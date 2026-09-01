#!/usr/bin/env python3
"""
run_esm2_score.py — ESM-2 pseudo-likelihood 기반 항체 서열 점수화

ProteinMPNN이 제안한 VH/VL 서열들을 ESM-2 언어 모델로 재평가합니다.
ESM-2 pseudo-log-likelihood (PLL)이 높을수록 자연 단백질에 가까운 서열입니다.

용도:
  1. ProteinMPNN top-96 후보 추가 필터링
  2. 휴머나이제이션 변이체 점수 비교
  3. CDR 잔기별 mutational effect 예측

사용법:
    python src/run_esm2_score.py \
        --fasta results/gpc3_top96.fasta \
        --out_dir results/esm2_scores \
        --model esm2_t33_650M_UR50D \
        --mode pll

    # 변이 효과 예측 (단일 서열 기준)
    python src/run_esm2_score.py \
        --fasta results/gpc3_top96.fasta \
        --out_dir results/esm2_scores \
        --mode mutational_scan \
        --target_regions CDR-H1,CDR-H2,CDR-H3,CDR-L1,CDR-L2,CDR-L3

설치:
    pip install fair-esm           # ESM-2 (Meta)
    pip install torch              # PyTorch (CUDA or Metal)

모델 크기:
    esm2_t6_8M_UR50D      — 8M  파라미터, ~30 MB  (CPU 빠름)
    esm2_t12_35M_UR50D    — 35M 파라미터, ~140 MB
    esm2_t30_150M_UR50D   — 150M파라미터, ~600 MB
    esm2_t33_650M_UR50D   — 650M파라미터, ~2.5 GB (권장, RTX 3060 Ti)
    esm2_t36_3B_UR50D     — 3B  파라미터, ~12 GB  (A100 이상)

출력:
    results/esm2_scores/
        scores.csv           — PLL, 정규화 PLL, 잔기별 점수
        mutational_scan/     — 각 CDR 위치별 AA 치환 효과 히트맵 데이터
        summary.json
"""

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Chothia CDR 정의 (VH/VL 잔기 번호, 1-based) ──────────────────────────────
# 실제 사용 시 AbNumber로 넘버링 후 매핑 권장
CDR_REGIONS_VH = {
    "CDR-H1": (26, 32),   # Chothia
    "CDR-H2": (52, 56),
    "CDR-H3": (95, 102),
}
CDR_REGIONS_VL = {
    "CDR-L1": (24, 34),
    "CDR-L2": (50, 56),
    "CDR-L3": (89, 97),
}

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


# ── 데이터 클래스 ──────────────────────────────────────────────────────────────
@dataclass
class ESM2Score:
    candidate_id: str
    chain: str           # "VH" or "VL"
    sequence: str
    pll: float           # pseudo-log-likelihood (sum)
    pll_per_residue: float  # PLL / length (정규화)
    length: int


# ── FASTA 유틸 ─────────────────────────────────────────────────────────────────
def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    current: Optional[str] = None
    buf: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current is not None:
                    seqs[current] = "".join(buf)
                current = line[1:]
                buf = []
            else:
                buf.append(line)
    if current is not None:
        seqs[current] = "".join(buf)
    return seqs


# ── ESM-2 로더 ────────────────────────────────────────────────────────────────
def load_esm2(model_name: str):
    """ESM-2 모델과 알파벳을 로드합니다."""
    try:
        import esm
        import torch
    except ImportError:
        log.error("fair-esm 미설치. 'pip install fair-esm' 실행")
        sys.exit(1)

    log.info(f"ESM-2 모델 로드: {model_name}")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()

    import torch
    if torch.cuda.is_available():
        model = model.cuda()
        log.info("  GPU (CUDA) 사용")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model = model.to("mps")
        log.info("  Apple Metal (MPS) 사용")
    else:
        log.info("  CPU 사용 (느림)")

    return model, alphabet


# ── PLL 계산 ──────────────────────────────────────────────────────────────────
def compute_pll(sequence: str, model, alphabet, batch_size: int = 1) -> tuple[float, list[float]]:
    """
    Masked marginal pseudo-log-likelihood 계산.

    각 위치를 순서대로 마스킹하고 ESM-2의 로짓에서 해당 아미노산 확률을 추출합니다.
    PLL = sum_i log P(x_i | x_{-i})

    Returns:
        (total_pll, per_residue_pll_list)
    """
    import torch
    import torch.nn.functional as F

    batch_converter = alphabet.get_batch_converter()
    data = [("seq", sequence)]
    _, _, tokens = batch_converter(data)

    device = next(model.parameters()).device
    tokens = tokens.to(device)

    L = len(sequence)
    per_residue = []

    with torch.no_grad():
        for i in range(L):
            # 위치 i 마스킹 (토큰 인덱스 1 = <mask>)
            masked = tokens.clone()
            masked[0, i + 1] = alphabet.mask_idx  # +1 for <cls>

            logits = model(masked, repr_layers=[], return_contacts=False)["logits"]
            # logits: [1, L+2, vocab]
            log_probs = F.log_softmax(logits[0, i + 1, :], dim=-1)

            aa_idx = alphabet.get_idx(sequence[i])
            per_residue.append(log_probs[aa_idx].item())

    total_pll = sum(per_residue)
    return total_pll, per_residue


# ── PLL 배치 계산 (효율적) ────────────────────────────────────────────────────
def compute_pll_batch(sequences: list[str], model, alphabet) -> list[tuple[float, list[float]]]:
    """여러 서열의 PLL을 순차적으로 계산합니다."""
    results = []
    for seq in sequences:
        pll, per_res = compute_pll(seq, model, alphabet)
        results.append((pll, per_res))
    return results


# ── 변이 효과 스캔 ─────────────────────────────────────────────────────────────
def mutational_scan(
    sequence: str,
    positions: list[int],  # 0-based
    model,
    alphabet,
    out_dir: Path,
    chain_id: str,
) -> dict:
    """
    지정된 위치에서 모든 AA 치환 효과를 계산합니다 (단일 마스킹).

    Returns:
        {pos: {aa: delta_pll}} — delta_pll = log P(mut) - log P(wt)
    """
    import torch
    import torch.nn.functional as F

    batch_converter = alphabet.get_batch_converter()
    data = [("seq", sequence)]
    _, _, tokens = batch_converter(data)
    device = next(model.parameters()).device
    tokens = tokens.to(device)

    scan_results: dict[int, dict[str, float]] = {}

    with torch.no_grad():
        for pos in positions:
            if pos >= len(sequence):
                continue
            wt_aa = sequence[pos]
            masked = tokens.clone()
            masked[0, pos + 1] = alphabet.mask_idx

            logits = model(masked, repr_layers=[], return_contacts=False)["logits"]
            log_probs = torch.nn.functional.log_softmax(
                logits[0, pos + 1, :], dim=-1
            )

            wt_idx = alphabet.get_idx(wt_aa)
            wt_logp = log_probs[wt_idx].item()

            pos_results: dict[str, float] = {}
            for mut_aa in AA_ALPHABET:
                mut_idx = alphabet.get_idx(mut_aa)
                if mut_idx < 0:
                    continue
                delta = log_probs[mut_idx].item() - wt_logp
                pos_results[mut_aa] = round(delta, 4)

            scan_results[pos] = pos_results

    # 히트맵 CSV 저장
    scan_dir = out_dir / "mutational_scan"
    scan_dir.mkdir(exist_ok=True)
    csv_path = scan_dir / f"{chain_id}_scan.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position_0based", "wt_aa"] + AA_ALPHABET)
        for pos, mutations in sorted(scan_results.items()):
            wt_aa = sequence[pos]
            row = [pos, wt_aa] + [mutations.get(aa, 0.0) for aa in AA_ALPHABET]
            writer.writerow(row)

    log.info(f"  변이 스캔 저장 → {csv_path}")
    return scan_results


# ── CDR 위치 추출 ──────────────────────────────────────────────────────────────
def get_cdr_positions(seq_len: int, chain_type: str) -> list[int]:
    """
    Chothia CDR 범위에서 0-based 인덱스 목록을 반환합니다.
    실제 넘버링과 다를 수 있으므로, 짧은 서열은 클램핑 처리.
    """
    if chain_type == "VH":
        regions = CDR_REGIONS_VH
    else:
        regions = CDR_REGIONS_VL

    positions = []
    for (start, end) in regions.values():
        for pos in range(start - 1, min(end, seq_len)):  # 0-based, clamp
            positions.append(pos)
    return sorted(set(positions))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="ESM-2 pseudo-likelihood 항체 서열 점수화")
    ap.add_argument("--fasta", required=True,
                    help="VH/VL 후보 서열 FASTA (헤더에 VH/VL 포함)")
    ap.add_argument("--out_dir", default="results/esm2_scores")
    ap.add_argument("--model", default="esm2_t33_650M_UR50D",
                    choices=["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D",
                             "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D",
                             "esm2_t36_3B_UR50D"],
                    help="ESM-2 모델 크기 (기본: 650M)")
    ap.add_argument("--mode", choices=["pll", "mutational_scan", "both"],
                    default="both",
                    help="실행 모드: pll=점수만, mutational_scan=CDR 변이 효과, both=둘다")
    ap.add_argument("--top_k", type=int, default=10,
                    help="상위 K개 후보 출력")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 서열 로드
    seqs = read_fasta(Path(args.fasta))
    log.info(f"서열 로드: {len(seqs)}개")

    # VH/VL 분류
    vh_seqs: dict[str, str] = {}
    vl_seqs: dict[str, str] = {}
    for header, seq in seqs.items():
        h = header.upper()
        cid = header.split("|")[-1] if "|" in header else header
        if "VH" in h or header.startswith("VH"):
            vh_seqs[cid] = seq
        elif "VL" in h or header.startswith("VL"):
            vl_seqs[cid] = seq
        else:
            # 구분 불가 → VH로 처리
            vh_seqs[cid] = seq

    log.info(f"  VH: {len(vh_seqs)}개  VL: {len(vl_seqs)}개")

    # ESM-2 로드
    model, alphabet = load_esm2(args.model)

    all_scores: list[ESM2Score] = []

    # PLL 계산
    if args.mode in ("pll", "both"):
        log.info("── PLL 계산 시작 ──")
        for chain_type, chain_seqs in [("VH", vh_seqs), ("VL", vl_seqs)]:
            for cid, seq in chain_seqs.items():
                log.info(f"  {chain_type} {cid} ({len(seq)} aa)")
                pll, per_res = compute_pll(seq, model, alphabet)
                score = ESM2Score(
                    candidate_id=cid,
                    chain=chain_type,
                    sequence=seq,
                    pll=round(pll, 4),
                    pll_per_residue=round(pll / max(len(seq), 1), 4),
                    length=len(seq),
                )
                all_scores.append(score)
                log.info(f"    PLL={pll:.2f}  PLL/res={pll/len(seq):.4f}")

        # 정규화 PLL 기준 랭킹
        all_scores.sort(key=lambda s: s.pll_per_residue, reverse=True)

        scores_csv = out_dir / "scores.csv"
        with open(scores_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(all_scores[0]).keys()))
            writer.writeheader()
            for s in all_scores:
                row = asdict(s)
                row.pop("sequence")  # 서열은 별도 저장
                writer.writerow(row)
        log.info(f"PLL 점수 저장 → {scores_csv}")

        log.info(f"\n상위 {args.top_k} (PLL/잔기 기준):")
        for rank, s in enumerate(all_scores[:args.top_k]):
            log.info(f"  {rank+1}. [{s.chain}] {s.candidate_id:30s}"
                     f"  PLL/res={s.pll_per_residue:.4f}  L={s.length}")

    # 변이 효과 스캔
    if args.mode in ("mutational_scan", "both"):
        log.info("\n── CDR 변이 효과 스캔 ──")
        # 상위 후보만 스캔 (시간 절약)
        top_vh = sorted(vh_seqs.items(),
                        key=lambda x: next((s.pll_per_residue for s in all_scores
                                           if s.candidate_id == x[0] and s.chain == "VH"), -999),
                        reverse=True)[:3]
        top_vl = sorted(vl_seqs.items(),
                        key=lambda x: next((s.pll_per_residue for s in all_scores
                                           if s.candidate_id == x[0] and s.chain == "VL"), -999),
                        reverse=True)[:3]

        for chain_type, top_list in [("VH", top_vh), ("VL", top_vl)]:
            for cid, seq in top_list:
                log.info(f"  스캔: {chain_type} {cid}")
                cdr_positions = get_cdr_positions(len(seq), chain_type)
                mutational_scan(seq, cdr_positions, model, alphabet,
                                out_dir, f"{cid}_{chain_type}")

    # 요약
    summary = {
        "model": args.model,
        "n_sequences": len(all_scores),
        "mode": args.mode,
        "top_k": [asdict(s) for s in all_scores[:args.top_k]],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\n완료. 출력 디렉토리: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
