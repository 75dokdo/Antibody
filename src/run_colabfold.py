#!/usr/bin/env python3
"""
ColabFold 복합체 입력 준비 및 실행 가이드 생성
FAP ECD (1Z68 chain A res 51-390) + scFv Top 5

Input:  fap_design/candidates/top5.fasta (scFv 서열)
Output: fap_design/colabfold/<id>.fasta   (multimer FASTA)
        fap_design/colabfold/run_colabfold.sh (실행 스크립트)

FAP ECD 서열: UniProt Q12884 (human FAP), aa 26–760 (세포외 도메인)
  – 1Z68 구조: chain A, SEQRES 위치 약 51-390 (β-propeller blade 1-7)
  – Blade 6-7 (epitope 타깃): 308–361 잔기
"""

import argparse
import json
import os
import sys

# ─── FAP ECD 서열 ──────────────────────────────────────────────────────────────
# Human FAP (Q12884) extracellular domain residues 26-760
# For ColabFold complex we use the prolyl endopeptidase domain + propeller:
# Approx residues 51-390 from 1Z68 structure (blade 1-7 β-propeller)
# Source: UniProt Q12884 / PDB 1Z68 chain A
FAP_ECD_1Z68 = (
    "GQQSAGSPFPVNFTQKNWLSLAAQRALFQTLQKASSDSGIYMVNQTPQGSDAGVLVYSGVIESGSIRLSWVQHNP"
    "YFDVIAHHPQKLAFSTEKSTSSPQAKLNVTPQLEEWRQTLRSHIQFNYGTSTTDATLKPGSQTIEVNLASSDVTP"
    "DPETLLPNSNLKNLQSTKYSQDKFQNLSQMDTLSAEYQAHSGKSVVTIDTDHFRLFSSSHQYVLVEHKSATTSFY"
    "EFAVGQSSMTQVNMKYTFQLSQNDTRVQMNDNPVISMRSGYFMSATLPKDIDVLPIQKTSALNFKTYNKYVLEFY"
    "TPEETFHKAAKMGQINLQSNYQILALDHTVKPSKLDSVFSSALSFIHQAQFDHILSLFNHYEAYTLR"
)

# Linker residue numbers for Blade 6-7 (FAP epitope target)
BLADE67_RESIDUES = list(range(308, 362))  # 308-361 inclusive


def make_multimer_fasta(fap_ecd: str, scfv_seq: str, cand_id: str) -> str:
    """
    ColabFold multimer FASTA format:
      - Two chains separated by ':' in a single sequence record
      OR separate records (depends on ColabFold version)
    We use the colon-separated format (LocalColabFold v1.5+)
    """
    # ColabFold multimer: chain A = FAP ECD, chain B = scFv
    combined = f"{fap_ecd}:{scfv_seq}"
    header = f">{cand_id}_FAP_ECD_scFv"
    return f"{header}\n{combined}\n"


def parse_fasta(fasta_path: str):
    """Parse FASTA file, return list of (header, seq)."""
    entries = []
    header, seq = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq)))
                header = line[1:]
                seq = []
            else:
                seq.append(line)
    if header is not None:
        entries.append((header, "".join(seq)))
    return entries


def parse_args():
    p = argparse.ArgumentParser(description="ColabFold input 준비 for FAP scFv Top 5")
    p.add_argument("--fasta", default="fap_design/candidates/top5.fasta")
    p.add_argument("--out_dir", default="fap_design/colabfold")
    p.add_argument("--fap_ecd", default=None,
                   help="FAP ECD 서열 파일 (없으면 내장 서열 사용)")
    p.add_argument("--num_recycles", type=int, default=3,
                   help="AlphaFold2 recycles (default 3, 최고 품질은 20)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # FAP ECD 서열
    if args.fap_ecd and os.path.exists(args.fap_ecd):
        with open(args.fap_ecd) as f:
            lines = f.read().strip().splitlines()
        fap_ecd = "".join(l for l in lines if not l.startswith(">"))
        print(f"[FAP ECD] 외부 파일 사용: {args.fap_ecd} ({len(fap_ecd)} aa)")
    else:
        fap_ecd = FAP_ECD_1Z68
        print(f"[FAP ECD] 내장 서열 사용 (1Z68 기반, {len(fap_ecd)} aa)")

    # scFv FASTA 읽기
    entries = parse_fasta(args.fasta)
    print(f"[scFv] {len(entries)}개 후보 로드: {args.fasta}")

    # 복합체 FASTA 생성
    out_fastas = []
    for header, scfv_seq in entries:
        parts = header.split("_")
        cand_id = "_".join(parts[1:3]) if len(parts) >= 3 else header.replace(" ", "_")
        label = header.split("_")[1] if "_" in header else header  # e.g. FAP-scFv-12534

        fasta_content = make_multimer_fasta(fap_ecd, scfv_seq, cand_id)
        out_path = os.path.join(args.out_dir, f"{cand_id}.fasta")
        with open(out_path, "w") as f:
            f.write(fasta_content)
        out_fastas.append(out_path)
        print(f"  → {out_path} (FAP={len(fap_ecd)}aa + scFv={len(scfv_seq)}aa = {len(fap_ecd)+len(scfv_seq)}aa total)")

    # 배치 FASTA (all 5 in one file for LocalColabFold batch)
    batch_path = os.path.join(args.out_dir, "top5_FAP_complex_batch.fasta")
    with open(batch_path, "w") as f:
        for header, scfv_seq in entries:
            parts = header.split("_")
            cand_id = "_".join(parts[1:3]) if len(parts) >= 3 else header.replace(" ", "_")
            f.write(make_multimer_fasta(fap_ecd, scfv_seq, cand_id))
    print(f"\n[배치 FASTA] → {batch_path}")

    # ColabFold 실행 스크립트 생성
    run_sh = os.path.join(args.out_dir, "run_colabfold.sh")
    with open(run_sh, "w") as f:
        f.write(f"""#!/bin/bash
# ColabFold FAP ECD + scFv 복합체 구조 예측
# LocalColabFold v1.5+ 필요: https://github.com/YoshitakaMo/localcolabfold
# GPU 권장 (M5 Pro Metal 또는 CUDA)
#
# 설치: bash <(curl -fsSL https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh)
# 또는 conda install -c conda-forge colabfold

set -e

COLABFOLD_BIN=$(which colabfold_batch 2>/dev/null || echo "$HOME/localcolabfold/colabfold-conda/bin/colabfold_batch")

if [ ! -f "$COLABFOLD_BIN" ]; then
    echo "[ERROR] colabfold_batch 미설치. LocalColabFold 설치 후 재시도."
    echo "  설치: https://github.com/YoshitakaMo/localcolabfold"
    exit 1
fi

echo "[ColabFold] FAP ECD + scFv 복합체 예측 시작"
echo "  입력: {batch_path}"
echo "  출력: {args.out_dir}/results/"
echo ""

mkdir -p {args.out_dir}/results

# 배치 실행 (Top 5 한번에)
$COLABFOLD_BIN \\
    {batch_path} \\
    {args.out_dir}/results \\
    --num-recycle {args.num_recycles} \\
    --model-type alphafold2_multimer_v3 \\
    --use-gpu-relax \\
    --num-models 5 \\
    --rank ipTM

echo ""
echo "[완료] 결과: {args.out_dir}/results/"
echo ""
echo "다음 단계: python3 src/analyze_colabfold.py"
""")
    os.chmod(run_sh, 0o755)
    print(f"[실행 스크립트] → {run_sh}")

    # 평가 기준 요약
    print(f"""
=== ColabFold 평가 기준 ===
  ipTM ≥ 0.5  → 복합체 신뢰도 기준
  pTM  ≥ 0.6  → 전체 구조 신뢰도
  Blade 6-7 접촉 확인: FAP 잔기 {BLADE67_RESIDUES[0]}–{BLADE67_RESIDUES[-1]}
    E311, D313 (음성 전하) → CDR 양성 잔기 접촉
    R356, K360, F358 (blade 7) → CDR 방향족/소수성 접촉

실행 방법:
  cd {os.getcwd()}
  bash {run_sh}

또는 Google Colab (무료 T4):
  https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb
  - Input: top5_FAP_complex_batch.fasta 업로드
  - jobname: FAP_scFv_complex
  - num_recycles: 3 (빠른 스크리닝)
  - model_type: alphafold2_multimer_v3
""")

    # metadata JSON
    meta = {
        "fap_ecd_len": len(fap_ecd),
        "fap_ecd_source": "1Z68_chain_A_51-390",
        "blade67_residues": f"{BLADE67_RESIDUES[0]}-{BLADE67_RESIDUES[-1]}",
        "epitope_key_residues": ["E311", "D313", "R356", "K360", "F358"],
        "scfv_count": len(entries),
        "num_recycles": args.num_recycles,
        "model_type": "alphafold2_multimer_v3",
        "iptm_cutoff": 0.5,
        "batch_fasta": batch_path,
        "run_script": run_sh,
    }
    meta_path = os.path.join(args.out_dir, "colabfold_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[메타데이터] → {meta_path}")


if __name__ == "__main__":
    main()
