#!/usr/bin/env python3
"""
run_humanize.py — CDR Grafting 기반 항체 휴머나이제이션 파이프라인

마우스/래빗 항체를 인간화하는 표준 CDR grafting 워크플로우:

  1. ANARCI로 Chothia/IMGT 넘버링 → CDR 경계 자동 탐지
  2. IMGT/V-QUEST 또는 로컬 데이터베이스에서 최적 인간 germline 프레임워크 선택
  3. CDR을 인간 FR에 이식 (CDR grafting)
  4. 복귀변이(backmutation) 후보 탐지: 패킹/VH-VL 접촉/Vernier zone
  5. ESM-2 PLL로 각 변이체 점수화
  6. 상위 휴머나이제이션 변이체 FASTA 출력

사용법:
    python src/run_humanize.py \
        --vh_seq EVQLVESGG... \
        --vl_seq DIQMTQSPS... \
        --candidate_id cand_001 \
        --out_dir results/humanized \
        --n_backmut_combos 16

    # FASTA 파일 입력
    python src/run_humanize.py \
        --fasta results/gpc3_top96.fasta \
        --out_dir results/humanized \
        --top_candidates 10

설치:
    pip install anarci           # 항체 넘버링 (Chothia/IMGT/Kabat)
    pip install fair-esm         # ESM-2 (변이체 점수화)
    pip install biopython        # BLAST (germline 검색)

    # IMGT germline 데이터베이스 (로컬)
    # https://www.imgt.org/vquest/refseqh.html 에서 다운로드 후:
    # data/imgt_germlines_VH.fasta
    # data/imgt_germlines_VL.fasta

출력:
    results/humanized/
        <candidate_id>/
            humanized_variants.fasta   — 모든 변이체 서열
            scores.csv                 — ESM-2 PLL 랭킹
            grafting_report.json       — CDR/FR 경계, germline 선택 근거
            top1_VH.fasta              — 최고 점수 VH
            top1_VL.fasta              — 최고 점수 VL
"""

import argparse
import csv
import itertools
import json
import logging
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 인간 germline 데이터베이스 경로 (로컬) ──────────────────────────────────
GERMLINE_VH_FASTA = Path("data/imgt_germlines_VH.fasta")
GERMLINE_VL_FASTA = Path("data/imgt_germlines_VL.fasta")

# ── Chothia CDR 범위 (ANARCI 넘버 기준, 1-based) ─────────────────────────────
CHOTHIA_CDR_VH = {"H1": (26, 32), "H2": (52, 56), "H3": (95, 102)}
CHOTHIA_CDR_VL = {"L1": (24, 34), "L2": (50, 56), "L3": (89, 97)}

# ── Vernier zone (VH-VL 패킹에 관여하는 FR 잔기) ─────────────────────────────
# Foote & Winter 1992 정의
VERNIER_VH = {2, 27, 29, 30, 36, 47, 48, 49, 67, 69, 71, 73, 78, 94}
VERNIER_VL = {2, 36, 46, 48, 49, 51, 54, 55, 58, 71, 87}


# ── 데이터 클래스 ──────────────────────────────────────────────────────────────
@dataclass
class Numbering:
    """ANARCI 넘버링 결과"""
    numbered: list[tuple[int, str, str]]  # [(resnum, insertion_code, aa)]
    chain_type: str                        # "H" or "L"


@dataclass
class CDRSegment:
    name: str
    start_pos: int  # 1-based Chothia
    end_pos: int
    sequence: str


@dataclass
class HumanizedVariant:
    candidate_id: str
    chain_type: str          # "VH" or "VL"
    germline_id: str
    germline_identity: float # FR 동일성 (%)
    backmutations: list[str] # ["H71V", "H78A", ...]
    sequence: str
    pll: float
    pll_per_residue: float
    humanness_score: float   # 0-1, 높을수록 인간에 가까움


# ── FASTA 유틸 ────────────────────────────────────────────────────────────────
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


# ── ANARCI 넘버링 ─────────────────────────────────────────────────────────────
def number_sequence(sequence: str, chain_type: str, scheme: str = "chothia") -> Optional[Numbering]:
    """
    ANARCI로 항체 서열을 넘버링합니다.

    Returns None if ANARCI is not installed.
    """
    try:
        from anarci import anarci, run_anarci
    except ImportError:
        log.warning("ANARCI 미설치. 'pip install anarci'로 설치하세요.")
        return None

    try:
        results = anarci(
            [("seq", sequence)],
            scheme=scheme,
            output=False,
            assign_germline=False,
        )
        numbered_list, _, _ = results
        if not numbered_list or numbered_list[0] is None:
            return None

        numbered = numbered_list[0][0][0]  # [(resnum, insertion, aa)]
        ct = numbered_list[0][1][0][0]    # "H" or "L" or "K"
        return Numbering(numbered=numbered, chain_type=ct)
    except Exception as exc:
        log.warning(f"ANARCI 넘버링 실패: {exc}")
        return None


def extract_cdrs_frs(numbering: Numbering) -> tuple[dict[str, str], dict[str, str]]:
    """
    넘버링 결과에서 CDR과 FR 서열을 추출합니다.

    Returns:
        (cdrs, frs) — {"H1": "GYTFTSYW", ...}, {"FR1": "EVQLV...", ...}
    """
    is_heavy = numbering.chain_type == "H"
    cdr_ranges = CHOTHIA_CDR_VH if is_heavy else CHOTHIA_CDR_VL

    positions = [(n, ins, aa) for n, ins, aa in numbering.numbered if aa != "-"]

    cdrs: dict[str, str] = {}
    frs: dict[str, str] = {}

    cdr_names = list(cdr_ranges.keys())
    boundaries = [(None, None)] + [(s, e) for s, e in cdr_ranges.values()] + [(None, None)]

    for i, (cdr_name, (cdr_start, cdr_end)) in enumerate(cdr_ranges.items()):
        cdr_seq = "".join(aa for n, ins, aa in positions
                          if cdr_start <= n <= cdr_end)
        cdrs[cdr_name] = cdr_seq

    # FR 추출 (CDR 사이 및 앞뒤)
    all_cdr_pos = set()
    for start, end in cdr_ranges.values():
        all_cdr_pos.update(range(start, end + 1))

    fr_seq = "".join(aa for n, ins, aa in positions if n not in all_cdr_pos)
    frs["FR_all"] = fr_seq

    return cdrs, frs


# ── Germline 검색 ─────────────────────────────────────────────────────────────
def find_best_germline(
    sequence: str,
    chain_type: str,
    germline_db: Optional[Path] = None,
    top_n: int = 3,
) -> list[tuple[str, float, str]]:
    """
    FR 서열과 가장 유사한 인간 germline을 찾습니다.

    Args:
        sequence: VH 또는 VL 서열 (전체)
        chain_type: "VH" or "VL"
        germline_db: IMGT germline FASTA 경로 (없으면 내장 미니 DB 사용)

    Returns:
        [(germline_id, identity_percent, germline_sequence), ...]
    """
    db_path = germline_db or (GERMLINE_VH_FASTA if chain_type == "VH" else GERMLINE_VL_FASTA)

    if db_path.exists():
        germlines = read_fasta(db_path)
    else:
        log.warning(f"Germline DB 없음: {db_path}. 내장 미니 DB 사용.")
        # 최소 내장 germline (IGHV1-2*02, IGKV1-39*01 대표 서열)
        if chain_type == "VH":
            germlines = {
                "IGHV1-2*02": (
                    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGII"
                    "NPSGGSTSYAQKFQGRVTMTRDTSTSTVYMELSSLRSEDTAVYYCAR"
                ),
                "IGHV3-23*01": (
                    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS"
                    "AISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR"
                ),
                "IGHV4-34*01": (
                    "QVQLQESGPGLVKPSETLSLTCTVSGGSISSGYYWNWIRQPPGKGLEWIGS"
                    "IYYSGSTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCAR"
                ),
            }
        else:
            germlines = {
                "IGKV1-39*01": (
                    "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPD"
                    "RFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSPWTFGQGTKVEIK"
                ),
                "IGKV3-20*01": (
                    "EIVMTQSPATLSVSPGERATLSCRASQSVSSNLAWYQQKPGQAPRLLIYDASTRATGIPAR"
                    "FSGSGSGTEFTLTISSLQSEDFAVYYCQQYNRYPYTFGQGTKLEIK"
                ),
                "IGLV2-14*01": (
                    "QSALTQPASVSGSPGQSITISCTGTSSDVGSYNLVSWYQQHPGKAPKLMIYDVSNRPSGVS"
                    "NRFSGSKSGNTASLTISGLQAEDEADYYCSSYTSSSTRVFGGGTKLTVL"
                ),
            }

    # 단순 서열 동일성 계산 (Needleman-Wunsch 대신 빠른 근사)
    results = []
    for gid, gseq in germlines.items():
        min_len = min(len(sequence), len(gseq))
        matches = sum(a == b for a, b in zip(sequence[:min_len], gseq[:min_len]))
        identity = matches / max(len(gseq), 1) * 100
        results.append((gid, identity, gseq))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


# ── CDR Grafting ──────────────────────────────────────────────────────────────
def graft_cdrs(
    donor_seq: str,
    donor_numbering: Numbering,
    acceptor_germ_seq: str,
    chain_type: str,
) -> str:
    """
    Donor CDR을 acceptor (인간 germline) FR에 이식합니다.

    단순화된 위치 기반 이식 (ANARCI 넘버링 사용).
    """
    is_heavy = chain_type == "VH"
    cdr_ranges = CHOTHIA_CDR_VH if is_heavy else CHOTHIA_CDR_VL

    # Donor 넘버링에서 CDR 서열 추출
    donor_positions = {(n, ins): aa
                       for n, ins, aa in donor_numbering.numbered
                       if aa != "-"}

    cdr_seqs: dict[str, str] = {}
    for cdr_name, (start, end) in cdr_ranges.items():
        cdr_seq = "".join(donor_positions.get((n, " "), "")
                          for n in range(start, end + 1)
                          if (n, " ") in donor_positions)
        cdr_seqs[cdr_name] = cdr_seq

    # Acceptor germline 넘버링
    acceptor_numbering = number_sequence(acceptor_germ_seq, chain_type[1])
    if acceptor_numbering is None:
        # ANARCI 없으면 단순 치환 (CDR 경계 추정)
        log.warning("ANARCI 없음 — 위치 기반 근사 CDR 이식")
        return _approximate_graft(donor_seq, acceptor_germ_seq, is_heavy, cdr_ranges)

    # Acceptor에서 CDR 위치 치환
    acc_list = list(acceptor_numbering.numbered)
    acc_dict = {(n, ins): i for i, (n, ins, aa) in enumerate(acc_list)}

    result = [aa for n, ins, aa in acc_list]
    for cdr_name, (start, end) in cdr_ranges.items():
        donor_cdr = cdr_seqs.get(cdr_name, "")
        acc_cdr_positions = [(n, ins, i) for i, (n, ins, aa) in enumerate(acc_list)
                             if start <= n <= end and aa != "-"]
        for j, (cdr_aa) in enumerate(donor_cdr):
            if j < len(acc_cdr_positions):
                _, _, idx = acc_cdr_positions[j]
                result[idx] = cdr_aa

    grafted = "".join(aa for aa in result if aa != "-")
    return grafted


def _approximate_graft(donor: str, acceptor: str, is_heavy: bool,
                        cdr_ranges: dict) -> str:
    """ANARCI 없을 때 길이 기반 근사 CDR 이식."""
    # 간단히 Donor CDR 위치(Chothia 1-based)를 0-based로 치환
    result = list(acceptor)
    cdr_seqs: dict[str, str] = {}
    for cdr_name, (start, end) in cdr_ranges.items():
        cdr_seq = donor[start-1:end] if end <= len(donor) else donor[start-1:]
        cdr_seqs[cdr_name] = cdr_seq

    for cdr_name, (start, end) in cdr_ranges.items():
        cdr = cdr_seqs[cdr_name]
        s, e = start - 1, min(end, len(result))
        for j, aa in enumerate(cdr):
            if s + j < e and s + j < len(result):
                result[s + j] = aa

    return "".join(result)


# ── 복귀변이 후보 탐지 ─────────────────────────────────────────────────────────
def find_backmutation_candidates(
    donor_numbering: Numbering,
    acceptor_seq: str,
    chain_type: str,
) -> list[tuple[int, str, str]]:
    """
    Vernier zone 및 CDR 경계 잔기에서 복귀변이 후보를 탐지합니다.

    Returns:
        [(position_1based, donor_aa, acceptor_aa), ...]
    """
    is_heavy = chain_type == "VH"
    vernier = VERNIER_VH if is_heavy else VERNIER_VL

    donor_map = {n: aa for n, ins, aa in donor_numbering.numbered if aa != "-"}
    acceptor_numbering = number_sequence(acceptor_seq, chain_type[1])
    if acceptor_numbering is None:
        return []

    acc_map = {n: aa for n, ins, aa in acceptor_numbering.numbered if aa != "-"}

    candidates = []
    for pos in sorted(vernier):
        d_aa = donor_map.get(pos, "-")
        a_aa = acc_map.get(pos, "-")
        if d_aa != a_aa and d_aa != "-" and a_aa != "-":
            candidates.append((pos, d_aa, a_aa))

    return candidates


# ── ESM-2 PLL 계산 (간소화 — run_esm2_score.py 재활용) ───────────────────────
def esm2_pll(sequence: str, model, alphabet) -> float:
    """ESM-2 pseudo-log-likelihood 합산값."""
    import torch
    import torch.nn.functional as F

    bc = alphabet.get_batch_converter()
    _, _, tokens = bc([("s", sequence)])
    device = next(model.parameters()).device
    tokens = tokens.to(device)

    L = len(sequence)
    total = 0.0
    with torch.no_grad():
        for i in range(L):
            masked = tokens.clone()
            masked[0, i + 1] = alphabet.mask_idx
            logits = model(masked, repr_layers=[], return_contacts=False)["logits"]
            lp = F.log_softmax(logits[0, i + 1, :], dim=-1)
            total += lp[alphabet.get_idx(sequence[i])].item()
    return total


def load_esm2(model_name: str = "esm2_t12_35M_UR50D"):
    """작은 ESM-2 모델 로드 (휴머나이제이션용 — 35M이 속도/정확도 균형 최적)."""
    try:
        import esm
        import torch
    except ImportError:
        return None, None

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()

    import torch
    if torch.cuda.is_available():
        model = model.cuda()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model = model.to("mps")
    return model, alphabet


# ── 인간화 파이프라인 ──────────────────────────────────────────────────────────
def humanize_sequence(
    vh_seq: str,
    vl_seq: str,
    candidate_id: str,
    out_dir: Path,
    n_backmut_combos: int = 16,
    esm_model_name: str = "esm2_t12_35M_UR50D",
) -> list[HumanizedVariant]:
    """
    단일 VH/VL 쌍에 대해 휴머나이제이션을 수행합니다.

    단계:
    1. ANARCI 넘버링 (CDR 자동 탐지)
    2. 최적 인간 germline 검색 (FR 동일성)
    3. CDR grafting (donor CDR → 인간 FR)
    4. Vernier zone 복귀변이 조합 탐색
    5. ESM-2 PLL로 변이체 점수화
    """
    cand_dir = out_dir / candidate_id
    cand_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "candidate_id": candidate_id,
        "vh_len": len(vh_seq),
        "vl_len": len(vl_seq),
        "steps": []
    }

    # 1. 넘버링
    log.info(f"[{candidate_id}] ANARCI 넘버링")
    vh_numbering = number_sequence(vh_seq, "H")
    vl_numbering = number_sequence(vl_seq, "L")

    if vh_numbering is None or vl_numbering is None:
        log.warning(f"  ANARCI 없음 — 위치 기반 근사 모드로 전환")
        # 넘버링 없이도 계속 진행 (근사 이식)

    # 2. Germline 검색
    log.info(f"[{candidate_id}] 인간 germline 검색")
    vh_germlines = find_best_germline(vh_seq, "VH")
    vl_germlines = find_best_germline(vl_seq, "VL")
    best_vh_germ_id, best_vh_fr_id, best_vh_germ_seq = vh_germlines[0]
    best_vl_germ_id, best_vl_fr_id, best_vl_germ_seq = vl_germlines[0]

    log.info(f"  VH 최적 germline: {best_vh_germ_id} ({best_vh_fr_id:.1f}% 동일성)")
    log.info(f"  VL 최적 germline: {best_vl_germ_id} ({best_vl_fr_id:.1f}% 동일성)")
    report["steps"].append({
        "step": "germline_selection",
        "vh": {"id": best_vh_germ_id, "identity": best_vh_fr_id},
        "vl": {"id": best_vl_germ_id, "identity": best_vl_fr_id},
    })

    # 3. CDR grafting
    log.info(f"[{candidate_id}] CDR grafting")
    if vh_numbering:
        grafted_vh = graft_cdrs(vh_seq, vh_numbering, best_vh_germ_seq, "VH")
    else:
        grafted_vh = _approximate_graft(vh_seq, best_vh_germ_seq, True, CHOTHIA_CDR_VH)

    if vl_numbering:
        grafted_vl = graft_cdrs(vl_seq, vl_numbering, best_vl_germ_seq, "VL")
    else:
        grafted_vl = _approximate_graft(vl_seq, best_vl_germ_seq, False, CHOTHIA_CDR_VL)

    report["steps"].append({
        "step": "cdr_grafting",
        "grafted_vh_len": len(grafted_vh),
        "grafted_vl_len": len(grafted_vl),
    })

    # 4. 복귀변이 후보
    log.info(f"[{candidate_id}] Vernier zone 복귀변이 후보 탐지")
    vh_backmut_candidates: list[tuple[int, str, str]] = []
    vl_backmut_candidates: list[tuple[int, str, str]] = []

    if vh_numbering:
        vh_backmut_candidates = find_backmutation_candidates(
            vh_numbering, grafted_vh, "VH")
    if vl_numbering:
        vl_backmut_candidates = find_backmutation_candidates(
            vl_numbering, grafted_vl, "VL")

    log.info(f"  VH 복귀변이 후보: {len(vh_backmut_candidates)}개")
    log.info(f"  VL 복귀변이 후보: {len(vl_backmut_candidates)}개")
    report["steps"].append({
        "step": "backmutation_candidates",
        "vh": [{"pos": p, "donor": d, "acceptor": a}
               for p, d, a in vh_backmut_candidates],
        "vl": [{"pos": p, "donor": d, "acceptor": a}
               for p, d, a in vl_backmut_candidates],
    })

    # 5. ESM-2 로드
    log.info(f"[{candidate_id}] ESM-2 로드 ({esm_model_name})")
    model, alphabet = load_esm2(esm_model_name)

    # 6. 변이체 조합 생성 및 점수화
    all_variants: list[HumanizedVariant] = []

    # 기본 grafted (복귀변이 없음)
    base_variants = [
        ("VH", grafted_vh, best_vh_germ_id, best_vh_fr_id, []),
        ("VL", grafted_vl, best_vl_germ_id, best_vl_fr_id, []),
    ]

    for chain_type, base_seq, germ_id, germ_id_pct, backmuts in base_variants:
        if model is not None and alphabet is not None:
            pll = esm2_pll(base_seq, model, alphabet)
            pll_per_res = pll / max(len(base_seq), 1)
        else:
            pll = 0.0
            pll_per_res = 0.0

        # 인간화 점수 = FR 동일성 (복귀변이 없을 때 최대)
        humanness = germ_id_pct / 100.0

        all_variants.append(HumanizedVariant(
            candidate_id=candidate_id,
            chain_type=chain_type,
            germline_id=germ_id,
            germline_identity=germ_id_pct,
            backmutations=backmuts,
            sequence=base_seq,
            pll=round(pll, 3),
            pll_per_residue=round(pll_per_res, 5),
            humanness_score=round(humanness, 3),
        ))

    # 복귀변이 조합 탐색 (VH만, 상위 n_backmut_combos 조합)
    if vh_backmut_candidates:
        # PowerSet의 상위 조합 (최대 4개 복귀변이씩)
        combo_count = 0
        for r in range(1, min(5, len(vh_backmut_candidates) + 1)):
            for combo in itertools.combinations(vh_backmut_candidates, r):
                if combo_count >= n_backmut_combos:
                    break

                mut_seq = list(grafted_vh)
                mut_labels = []
                for pos, donor_aa, acc_aa in combo:
                    # 위치 pos(1-based Chothia) → 0-based 근사
                    idx = pos - 1
                    if 0 <= idx < len(mut_seq):
                        mut_seq[idx] = donor_aa
                        mut_labels.append(f"H{pos}{donor_aa}")

                mut_seq_str = "".join(mut_seq)
                if model is not None and alphabet is not None:
                    pll = esm2_pll(mut_seq_str, model, alphabet)
                    pll_per_res = pll / max(len(mut_seq_str), 1)
                else:
                    pll = 0.0
                    pll_per_res = 0.0

                humanness = (best_vh_fr_id - len(combo) * 2) / 100.0

                all_variants.append(HumanizedVariant(
                    candidate_id=candidate_id,
                    chain_type="VH",
                    germline_id=best_vh_germ_id,
                    germline_identity=best_vh_fr_id,
                    backmutations=mut_labels,
                    sequence=mut_seq_str,
                    pll=round(pll, 3),
                    pll_per_residue=round(pll_per_res, 5),
                    humanness_score=round(max(0.0, humanness), 3),
                ))
                combo_count += 1

            if combo_count >= n_backmut_combos:
                break

    # PLL/잔기 기준 랭킹
    all_variants.sort(key=lambda v: v.pll_per_residue, reverse=True)

    # 저장
    fasta_path = cand_dir / "humanized_variants.fasta"
    with open(fasta_path, "w") as f:
        for i, v in enumerate(all_variants):
            bmuts = ",".join(v.backmutations) if v.backmutations else "none"
            f.write(f">{v.candidate_id}_{v.chain_type}_rank{i+1:02d}"
                    f"|{v.germline_id}|backmut={bmuts}"
                    f"|pll_per_res={v.pll_per_residue:.5f}"
                    f"|humanness={v.humanness_score:.3f}\n")
            f.write(v.sequence + "\n")

    scores_csv = cand_dir / "scores.csv"
    with open(scores_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_variants[0]).keys()))
        writer.writeheader()
        for v in all_variants:
            writer.writerow(asdict(v))

    report["n_variants"] = len(all_variants)
    report["top1_VH"] = asdict(next((v for v in all_variants if v.chain_type == "VH"), None) or HumanizedVariant("", "", "", 0, [], "", 0, 0, 0))
    report["top1_VL"] = asdict(next((v for v in all_variants if v.chain_type == "VL"), None) or HumanizedVariant("", "", "", 0, [], "", 0, 0, 0))

    with open(cand_dir / "grafting_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # 최고 점수 VH/VL 개별 저장
    top_vh = next((v for v in all_variants if v.chain_type == "VH"), None)
    top_vl = next((v for v in all_variants if v.chain_type == "VL"), None)
    if top_vh:
        with open(cand_dir / "top1_VH.fasta", "w") as f:
            f.write(f">VH|{candidate_id}|humanized\n{top_vh.sequence}\n")
    if top_vl:
        with open(cand_dir / "top1_VL.fasta", "w") as f:
            f.write(f">VL|{candidate_id}|humanized\n{top_vl.sequence}\n")

    log.info(f"[{candidate_id}] 변이체 {len(all_variants)}개 생성 → {cand_dir}")
    return all_variants


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="CDR Grafting 항체 휴머나이제이션")

    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--fasta",
                             help="후보 VH/VL FASTA 파일 (헤더에 VH/VL 포함)")
    input_group.add_argument("--vh_seq", help="VH 서열 직접 입력")

    ap.add_argument("--vl_seq", help="VL 서열 직접 입력 (--vh_seq와 함께)")
    ap.add_argument("--candidate_id", default="candidate",
                    help="후보 ID (직접 입력 시)")
    ap.add_argument("--out_dir", default="results/humanized")
    ap.add_argument("--top_candidates", type=int, default=10,
                    help="FASTA 입력 시 처리할 상위 N개 후보")
    ap.add_argument("--n_backmut_combos", type=int, default=16,
                    help="복귀변이 조합 탐색 수 (기본 16)")
    ap.add_argument("--esm_model", default="esm2_t12_35M_UR50D",
                    choices=["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D",
                             "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"],
                    help="ESM-2 모델 (기본: 35M — 속도/정확도 균형)")
    ap.add_argument("--germline_vh_db", type=Path,
                    help="IMGT VH germline FASTA (없으면 내장 미니 DB)")
    ap.add_argument("--germline_vl_db", type=Path,
                    help="IMGT VL germline FASTA (없으면 내장 미니 DB)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Germline DB 경로 오버라이드
    if args.germline_vh_db:
        global GERMLINE_VH_FASTA
        GERMLINE_VH_FASTA = args.germline_vh_db
    if args.germline_vl_db:
        global GERMLINE_VL_FASTA
        GERMLINE_VL_FASTA = args.germline_vl_db

    # 입력 처리
    if args.vh_seq:
        if not args.vl_seq:
            ap.error("--vl_seq 도 입력해야 합니다")
        pairs = [(args.candidate_id, args.vh_seq, args.vl_seq)]
    else:
        seqs = read_fasta(Path(args.fasta))
        vh_map: dict[str, str] = {}
        vl_map: dict[str, str] = {}
        for header, seq in seqs.items():
            h = header.upper()
            cid = header.split("|")[-1] if "|" in header else header
            if "VH" in h:
                vh_map[cid] = seq
            elif "VL" in h:
                vl_map[cid] = seq

        pairs = [(cid, vh_map[cid], vl_map[cid])
                 for cid in list(vh_map.keys())[:args.top_candidates]
                 if cid in vl_map]
        log.info(f"처리 후보: {len(pairs)}쌍")

    all_results: list[dict] = []
    for cid, vh, vl in pairs:
        variants = humanize_sequence(
            vh, vl, cid, out_dir,
            n_backmut_combos=args.n_backmut_combos,
            esm_model_name=args.esm_model,
        )
        top_vh = next((v for v in variants if v.chain_type == "VH"), None)
        top_vl = next((v for v in variants if v.chain_type == "VL"), None)
        all_results.append({
            "candidate_id": cid,
            "top_vh_pll_per_res": top_vh.pll_per_residue if top_vh else None,
            "top_vh_humanness": top_vh.humanness_score if top_vh else None,
            "top_vl_pll_per_res": top_vl.pll_per_residue if top_vl else None,
            "top_vl_humanness": top_vl.humanness_score if top_vl else None,
            "n_variants": len(variants),
        })

    # 전체 요약
    summary_csv = out_dir / "humanization_summary.csv"
    if all_results:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        log.info(f"요약 저장 → {summary_csv}")

    log.info(f"\n✓ 휴머나이제이션 완료: {len(pairs)}개 후보 처리")
    return 0


if __name__ == "__main__":
    sys.exit(main())
