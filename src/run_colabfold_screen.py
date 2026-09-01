#!/usr/bin/env python3
"""
run_colabfold_screen.py — ColabFold (AlphaFold2-Multimer) 항체-항원 복합체 구조 예측

VH/VL 후보를 GPC3 항원과 함께 ColabFold로 구조 예측하고,
pDockQ / ipTM / pTM 점수로 랭킹합니다.

사용법:
    # 로컬 ColabFold (GPU 권장)
    python src/run_colabfold_screen.py \
        --fasta_dir results/chai1_inputs \
        --antigen_fasta data/gpc3_mature.fasta \
        --out_dir results/colabfold_screen \
        --top_k 10

    # colabfold_batch 직접 호출 (명령행 래퍼)
    python src/run_colabfold_screen.py \
        --fasta_dir results/chai1_inputs \
        --antigen_fasta data/gpc3_mature.fasta \
        --out_dir results/colabfold_screen \
        --mode cli          # 'api' | 'cli'
        --colabfold_bin $(which colabfold_batch)

설치:
    # GPU 환경 (RTX 3060 Ti / CUDA)
    pip install "colabfold[cuda]" --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    # CPU / Apple Metal (M5) — 느리지만 작동
    pip install "colabfold[cpu]"

    # 또는 로컬 mmseqs2 서버 없이 서버 MSA 사용
    colabfold_batch --help

출력:
    results/colabfold_screen/
        scores.csv          — 전체 후보 랭킹 (ipTM, pDockQ, pTM)
        top_k/              — 상위-K PDB 구조
        inputs/             — ColabFold용 FASTA (antigen:VH:VL 형식)
        raw/<candidate>/    — ColabFold 원시 출력 (log, PDB, json)

ColabFold FASTA 포맷:
    >header
    ANTIGEN_SEQ:VH_SEQ:VL_SEQ
    (콜론으로 체인 구분 — ColabFold multimer 형식)
"""

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
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

# ── GPC3 에피토프 (참고용 — ColabFold는 구속 없음, 점수로만 평가) ──────────────
HOTSPOT_MATURE = [267, 270, 273, 274, 277, 372, 373, 375, 376, 379, 380]


# ── 데이터 클래스 ──────────────────────────────────────────────────────────────
@dataclass
class ColabFoldScore:
    candidate_id: str
    iptm: float          # interface pTM (체인간 신뢰도)
    ptm: float           # 전체 pTM
    pdockq: float        # pDockQ (복합체 도킹 품질)
    plddt_mean: float    # 평균 pLDDT
    plddt_interface: float  # 인터페이스 잔기 평균 pLDDT
    model_rank: int      # ColabFold 내부 모델 랭크 (1-5)
    fasta_path: str


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


def write_colabfold_fasta(antigen_seq: str, vh_seq: str, vl_seq: str,
                           out_path: Path, candidate_id: str) -> None:
    """
    ColabFold multimer 포맷: 체인을 콜론(:)으로 구분.
    순서: 항원 | VH | VL
    """
    combined = f"{antigen_seq}:{vh_seq}:{vl_seq}"
    with open(out_path, "w") as f:
        f.write(f">{candidate_id}\n{combined}\n")


# ── pDockQ 계산 ────────────────────────────────────────────────────────────────
def compute_pdockq(pdb_path: Path, interface_dist_cutoff: float = 8.0) -> tuple[float, float]:
    """
    pDockQ를 PDB 파일에서 직접 계산합니다 (Bryant et al. 2022).

    pDockQ = 0.724 / (1 + exp(-0.052*(x - 152.611))) + 0.018
    where x = interface_contacts * mean_interface_plddt

    Returns:
        (pdockq, mean_interface_plddt)
    """
    try:
        import numpy as np

        atoms: list[dict] = []
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                try:
                    chain = line[21]
                    resnum = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    bfactor = float(line[60:66])  # pLDDT stored in B-factor
                    atoms.append({"chain": chain, "res": resnum,
                                  "xyz": np.array([x, y, z]), "plddt": bfactor})
                except ValueError:
                    continue

        if not atoms:
            return 0.0, 0.0

        # 체인 분류 (첫 번째 = 항원, 나머지 = 항체)
        chains = list(dict.fromkeys(a["chain"] for a in atoms))
        if len(chains) < 2:
            return 0.0, 0.0

        antigen_chain = chains[0]
        ab_chains = set(chains[1:])

        ag_atoms = [a for a in atoms if a["chain"] == antigen_chain]
        ab_atoms = [a for a in atoms if a["chain"] in ab_chains]

        ag_xyz = np.array([a["xyz"] for a in ag_atoms])
        ab_xyz = np.array([a["xyz"] for a in ab_atoms])

        # 인터페이스 원자 탐색 (cutoff 이내)
        from scipy.spatial import cKDTree
        tree = cKDTree(ag_xyz)
        contact_indices = tree.query_ball_point(ab_xyz, r=interface_dist_cutoff)

        interface_ag_idx = set()
        interface_ab_idx = set()
        for ab_i, ag_contacts in enumerate(contact_indices):
            if ag_contacts:
                interface_ag_idx.update(ag_contacts)
                interface_ab_idx.add(ab_i)

        n_contacts = len(interface_ag_idx) + len(interface_ab_idx)
        if n_contacts == 0:
            return 0.0, 0.0

        interface_plddt = (
            np.mean([ag_atoms[i]["plddt"] for i in interface_ag_idx]) +
            np.mean([ab_atoms[i]["plddt"] for i in interface_ab_idx])
        ) / 2.0

        x = n_contacts * interface_plddt
        pdockq = 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018
        return float(pdockq), float(interface_plddt)

    except ImportError:
        log.warning("scipy not installed; pDockQ set to 0. Run: pip install scipy")
        return 0.0, 0.0
    except Exception as exc:
        log.warning(f"pDockQ 계산 실패: {exc}")
        return 0.0, 0.0


# ── ColabFold JSON 파싱 ────────────────────────────────────────────────────────
def parse_colabfold_scores(raw_dir: Path, candidate_id: str) -> Optional[ColabFoldScore]:
    """
    ColabFold 출력 디렉토리에서 최고 모델 점수를 파싱합니다.
    ColabFold는 <candidate_id>_scores_rank_001_*.json 형태로 저장합니다.
    """
    # 점수 JSON 파일 탐색
    score_files = sorted(raw_dir.glob(f"{candidate_id}_scores_rank_*.json"))
    if not score_files:
        # 다른 패턴도 시도
        score_files = sorted(raw_dir.glob("*_scores_rank_*.json"))

    if not score_files:
        log.warning(f"  점수 파일 없음: {raw_dir}")
        return None

    best_score_file = score_files[0]  # rank_001이 최고
    with open(best_score_file) as f:
        scores = json.load(f)

    iptm = float(scores.get("iptm", scores.get("interface_ptm", 0.0)))
    ptm = float(scores.get("ptm", 0.0))
    plddt_arr = scores.get("plddt", [])
    plddt_mean = float(np.mean(plddt_arr)) if plddt_arr else 0.0

    # 대응하는 PDB 찾기
    pdb_files = sorted(raw_dir.glob(f"{candidate_id}_unrelaxed_rank_001_*.pdb"))
    if not pdb_files:
        pdb_files = sorted(raw_dir.glob("*_unrelaxed_rank_001_*.pdb"))

    pdockq = 0.0
    plddt_interface = 0.0
    best_pdb = None
    if pdb_files:
        best_pdb = pdb_files[0]
        pdockq, plddt_interface = compute_pdockq(best_pdb)

    return ColabFoldScore(
        candidate_id=candidate_id,
        iptm=iptm,
        ptm=ptm,
        pdockq=pdockq,
        plddt_mean=plddt_mean,
        plddt_interface=plddt_interface,
        model_rank=1,
        fasta_path=str(raw_dir / f"{candidate_id}.fasta"),
    ), best_pdb


# ── CLI 모드: colabfold_batch 외부 호출 ────────────────────────────────────────
def run_colabfold_cli(
    fasta_path: Path,
    out_dir: Path,
    colabfold_bin: str = "colabfold_batch",
    n_models: int = 5,
    n_recycles: int = 3,
    use_amber: bool = False,
    msa_mode: str = "mmseqs2_uniref_env",
) -> bool:
    """
    colabfold_batch 명령어를 subprocess로 실행합니다.

    Returns:
        True if successful, False otherwise.
    """
    cmd = [
        colabfold_bin,
        str(fasta_path),
        str(out_dir),
        "--num-models", str(n_models),
        "--num-recycle", str(n_recycles),
        "--msa-mode", msa_mode,
        "--model-type", "alphafold2_multimer_v3",
        "--rank", "iptm",        # ipTM으로 모델 랭킹
    ]
    if use_amber:
        cmd.append("--amber")   # 릴렉세이션 (느리지만 더 정확)

    log.info(f"  실행: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=3600  # 1시간 타임아웃
    )

    if result.returncode != 0:
        log.error(f"  colabfold_batch 실패 (code={result.returncode})")
        log.error(f"  stderr: {result.stderr[-500:]}")
        return False

    return True


# ── API 모드: colabfold Python API 직접 호출 ──────────────────────────────────
def run_colabfold_api(
    fasta_path: Path,
    out_dir: Path,
    n_models: int = 5,
    n_recycles: int = 3,
) -> bool:
    """
    colabfold Python API를 직접 호출합니다.
    'colabfold' 패키지가 설치되어 있어야 합니다.
    """
    try:
        from colabfold.batch import get_queries, run
        from colabfold.download import download_alphafold_params
    except ImportError:
        log.error("colabfold 패키지 미설치. 'pip install colabfold[cuda]' 실행 후 재시도")
        return False

    queries, is_complex = get_queries(str(fasta_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        run(
            queries=queries,
            result_dir=str(out_dir),
            use_templates=False,
            num_relax=0,
            msa_mode="mmseqs2_uniref_env",
            model_type="alphafold2_multimer_v3",
            num_models=n_models,
            num_recycles=n_recycles,
            rank_by="iptm",
            pair_mode="unpaired_paired",
        )
        return True
    except Exception as exc:
        log.error(f"  API 호출 실패: {exc}")
        return False


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="ColabFold 항체-항원 복합체 배치 스크리닝")
    ap.add_argument("--fasta_dir", required=True,
                    help="후보별 FASTA 파일 디렉토리 (VH/VL 포함)")
    ap.add_argument("--antigen_fasta", required=True,
                    help="GPC3 항원 서열 FASTA 파일")
    ap.add_argument("--out_dir", default="results/colabfold_screen",
                    help="출력 디렉토리")
    ap.add_argument("--mode", choices=["cli", "api"], default="cli",
                    help="실행 방법: cli=colabfold_batch 외부 호출 (기본), api=Python API")
    ap.add_argument("--colabfold_bin", default="colabfold_batch",
                    help="colabfold_batch 실행 파일 경로 (cli 모드)")
    ap.add_argument("--n_models", type=int, default=5,
                    help="모델 수 (기본 5; 빠른 스크리닝은 2)")
    ap.add_argument("--n_recycles", type=int, default=3,
                    help="리사이클 수 (기본 3; 정밀 예측은 12)")
    ap.add_argument("--top_k", type=int, default=10,
                    help="상위 K개 구조 저장")
    ap.add_argument("--iptm_cutoff", type=float, default=0.5,
                    help="통과 기준 ipTM (기본 0.5)")
    ap.add_argument("--pdockq_cutoff", type=float, default=0.23,
                    help="통과 기준 pDockQ (기본 0.23; Elofsson 2022)")
    ap.add_argument("--use_amber", action="store_true",
                    help="Amber 릴렉세이션 적용 (cli 모드만, 느림)")
    args = ap.parse_args()

    fasta_dir = Path(args.fasta_dir)
    antigen_fasta = Path(args.antigen_fasta)
    out_dir = Path(args.out_dir)
    inputs_dir = out_dir / "inputs"
    raw_dir = out_dir / "raw"
    topk_dir = out_dir / "top_k"
    for d in [inputs_dir, raw_dir, topk_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 항원 서열 로드
    antigen_seq = next(iter(read_fasta(antigen_fasta).values()))
    log.info(f"항원: {len(antigen_seq)} 잔기")

    # 후보 FASTA 목록
    candidate_fastas = sorted(fasta_dir.glob("*.fasta")) + sorted(fasta_dir.glob("*.fa"))
    if not candidate_fastas:
        log.error(f"FASTA 파일 없음: {fasta_dir}")
        sys.exit(1)
    log.info(f"스크리닝 후보: {len(candidate_fastas)}개")

    results: list[ColabFoldScore] = []
    start_total = time.time()

    for i, cand_fasta in enumerate(candidate_fastas):
        candidate_id = cand_fasta.stem
        log.info(f"[{i+1}/{len(candidate_fastas)}] {candidate_id}")

        # VH/VL 파싱
        seqs = read_fasta(cand_fasta)
        vh_seq = vl_seq = None
        for header, seq in seqs.items():
            h = header.upper()
            if "VH" in h:
                vh_seq = seq
            elif "VL" in h:
                vl_seq = seq

        if not vh_seq or not vl_seq:
            log.warning(f"  VH/VL 파싱 실패, 건너뜀: {cand_fasta.name}")
            continue

        # ColabFold multimer FASTA 작성
        cf_fasta = inputs_dir / f"{candidate_id}.fasta"
        write_colabfold_fasta(antigen_seq, vh_seq, vl_seq, cf_fasta, candidate_id)

        cand_raw = raw_dir / candidate_id
        cand_raw.mkdir(exist_ok=True)

        # ColabFold 실행
        t0 = time.time()
        if args.mode == "cli":
            ok = run_colabfold_cli(
                cf_fasta, cand_raw,
                colabfold_bin=args.colabfold_bin,
                n_models=args.n_models,
                n_recycles=args.n_recycles,
                use_amber=args.use_amber,
            )
        else:
            ok = run_colabfold_api(cf_fasta, cand_raw,
                                   n_models=args.n_models,
                                   n_recycles=args.n_recycles)

        elapsed = time.time() - t0
        if not ok:
            log.warning(f"  {candidate_id}: 예측 실패 ({elapsed:.0f}s)")
            continue

        # 점수 파싱
        parsed = parse_colabfold_scores(cand_raw, candidate_id)
        if parsed is None:
            continue
        score, best_pdb = parsed

        log.info(
            f"  ipTM={score.iptm:.3f}  pTM={score.ptm:.3f}"
            f"  pDockQ={score.pdockq:.3f}  pLDDT={score.plddt_mean:.1f}"
            f"  ({elapsed:.0f}s)"
        )
        results.append(score)

        # 최고 모델 PDB 복사
        if best_pdb and best_pdb.exists():
            shutil.copy(best_pdb, cand_raw / f"{candidate_id}_best.pdb")

    if not results:
        log.error("유효한 결과 없음")
        sys.exit(1)

    # ── 랭킹: pDockQ 기준 (복합체 품질 지표로서 ipTM보다 더 신뢰성 높음) ──────
    results.sort(key=lambda r: r.pdockq, reverse=True)

    scores_csv = out_dir / "scores.csv"
    with open(scores_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    log.info(f"점수 저장 → {scores_csv}")

    # 상위 K 구조 복사
    for rank, r in enumerate(results[:args.top_k]):
        src = raw_dir / r.candidate_id / f"{r.candidate_id}_best.pdb"
        if src.exists():
            dst = topk_dir / f"rank{rank+1:02d}_{r.candidate_id}.pdb"
            shutil.copy(src, dst)

    # 요약
    passed = [r for r in results
              if r.iptm >= args.iptm_cutoff and r.pdockq >= args.pdockq_cutoff]
    log.info("=" * 60)
    log.info(f"스크리닝: {len(results)}개  |  통과: {len(passed)}개"
             f"  (ipTM≥{args.iptm_cutoff} AND pDockQ≥{args.pdockq_cutoff})")
    log.info(f"총 소요: {(time.time()-start_total)/60:.1f} min")
    log.info("상위 5 (pDockQ 기준):")
    for rank, r in enumerate(results[:5]):
        flag = "✓" if (r.iptm >= args.iptm_cutoff and r.pdockq >= args.pdockq_cutoff) else " "
        log.info(f"  {flag} {rank+1}. {r.candidate_id:30s}"
                 f"  pDockQ={r.pdockq:.3f}  ipTM={r.iptm:.3f}")

    summary = {
        "n_candidates": len(results),
        "n_passed": len(passed),
        "iptm_cutoff": args.iptm_cutoff,
        "pdockq_cutoff": args.pdockq_cutoff,
        "mode": args.mode,
        "n_models": args.n_models,
        "n_recycles": args.n_recycles,
        "top_k": [asdict(r) for r in results[:args.top_k]],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
