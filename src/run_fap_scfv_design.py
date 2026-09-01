#!/usr/bin/env python3
"""
FAP (Fibroblast Activation Protein) scFv 설계 파이프라인
프레임워크: 트라스투주맙 (Herceptin) VH/VL FR 재사용
CDR: FAP β-프로펠러 도메인 표적 de novo 설계
용도: CAR-T / CAR-NK 세포치료제

PDB 표적: 1Z68 (FAP 호모다이머, 2.5Å)
표적 에피토프: β-프로펠러 Blade 6-7 (FAP 특이적, DPP4 비공유)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ─── 트라스투주맙 (Trastuzumab / Herceptin) 시퀀스 ────────────────────────────
# 출처: UniProt P01857 (VH), P01834 (VL) / INN sequence
# 인간화 항체 (murine 4D5 → humanized), IGHV3-66 + IGKV1-39 기반

TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARI"
    "YPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGD"
    "GFYAMDYWGQGTLVTVSS"
)

TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYS"
    "ASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQG"
    "TKVEIK"
)

# 표준 scFv 링커 (G4S)×3 — CAR 구조에 최적
SCFV_LINKER_G4S3 = "GGGGSGGGGSGGGGS"
SCFV_LINKER_G4S4 = "GGGGSGGGGSGGGGSGGGGS"  # 더 유연한 링커 (대형 항원 접근 개선)

# ─── Chothia CDR 경계 (순번) ─────────────────────────────────────────────────
# VH Chothia CDR 위치 (1-indexed, 아래 추출 함수와 연동)
VH_CDR_CHOTHIA = {
    "H1": (26, 32),   # GFNIKDT  (trastuzumab)
    "H2": (52, 56),   # YPTNG
    "H3": (95, 102),  # SRWGGDGF (trastuzumab; 실제 H3 끝은 W103 앞까지)
}
# VL Chothia CDR 위치
VL_CDR_CHOTHIA = {
    "L1": (24, 34),   # RASQDVNTAVA
    "L2": (50, 56),   # SASFLYSGVPS → 50-56 = SASFLYSGVPS
    "L3": (89, 97),   # QQHYTTPPT
}

# ─── 트라스투주맙 프레임워크 추출 ─────────────────────────────────────────────
def extract_framework(seq: str, cdrs: dict) -> dict:
    """CDR 위치로부터 FR 서열 추출 (1-indexed Chothia)"""
    sorted_cdrs = sorted(cdrs.values())
    regions = {}
    prev = 0
    fr_idx = 1
    for start, end in sorted_cdrs:
        # FR (0-indexed slice)
        regions[f"FR{fr_idx}"] = seq[prev : start - 1]
        fr_idx += 1
        # CDR
        cdr_name = [k for k, v in cdrs.items() if v == (start, end)][0]
        regions[cdr_name] = seq[start - 1 : end]
        prev = end
    regions[f"FR{fr_idx}"] = seq[prev:]
    return regions


def get_trastuzumab_frameworks():
    """트라스투주맙 FR 서열 반환 (CDR 제거)"""
    vh_parts = extract_framework(TRASTUZUMAB_VH, VH_CDR_CHOTHIA)
    vl_parts = extract_framework(TRASTUZUMAB_VL, VL_CDR_CHOTHIA)

    vh_fr = {k: v for k, v in vh_parts.items() if k.startswith("FR")}
    vl_fr = {k: v for k, v in vl_parts.items() if k.startswith("FR")}
    vh_cdrs = {k: v for k, v in vh_parts.items() if not k.startswith("FR")}
    vl_cdrs = {k: v for k, v in vl_parts.items() if not k.startswith("FR")}

    return vh_fr, vl_fr, vh_cdrs, vl_cdrs


# ─── FAP 표적 정보 ─────────────────────────────────────────────────────────────
FAP_TARGET = {
    "name": "Fibroblast Activation Protein alpha (FAPα)",
    "uniprot": "Q12884",
    "pdb": "1Z68",        # 호모다이머 2.5Å
    "pdb_alt": ["5XBT", "5XBU", "6B2H"],
    "chain": "A",
    # ECD 전체 (신호펩타이드·TM 제거 후): 잔기 51-760
    "ecd_range": (51, 760),
    # β-프로펠러 도메인 (FAP 특이적, DPP4 비공유)
    "beta_propeller": (51, 390),
    # 표적 에피토프: Blade 6-7 루프 (가장 돌출, FAP 선택적)
    "epitope_hotspots": [
        308, 310, 312, 314,   # Blade 6 외향 루프
        355, 357, 359, 361,   # Blade 7 외향 루프 (1차 선택)
        205, 207, 209,        # Blade 4 루프 (2차)
    ],
    # 촉매 트리아드 (기능 차단 원할 경우 추가)
    "catalytic_triad": [624, 702, 734],  # Ser624, Asp702, His734
    # 다이머 계면 (호모다이머 파괴)
    "dimer_interface": [82, 83, 84, 86, 88, 415, 417],
}


# ─── scFv 조립 ────────────────────────────────────────────────────────────────
def build_scfv(vh: str, vl: str, orientation: str = "VH-VL",
               linker: str = SCFV_LINKER_G4S3) -> str:
    """scFv 구성 (VH-링커-VL 또는 VL-링커-VH)"""
    if orientation == "VH-VL":
        return f"{vh}{linker}{vl}"
    elif orientation == "VL-VH":
        return f"{vl}{linker}{vh}"
    else:
        raise ValueError(f"Unknown orientation: {orientation}")


def graft_cdrs_into_trastuzumab(vh_cdrs: dict, vl_cdrs: dict) -> tuple:
    """
    새로운 CDR을 트라스투주맙 FR에 이식
    vh_cdrs: {'H1': 'XXXXX', 'H2': 'XXXXX', 'H3': 'XXXXXXXX'}
    vl_cdrs: {'L1': 'XXXXXXXXXX', 'L2': 'XXXXXXX', 'L3': 'XXXXXXXXX'}
    """
    vh_fr, vl_fr, _, _ = get_trastuzumab_frameworks()

    # VH 조립: FR1-CDR-H1-FR2-CDR-H2-FR3-CDR-H3-FR4
    vh = (vh_fr["FR1"] + vh_cdrs["H1"] + vh_fr["FR2"] +
          vh_cdrs["H2"] + vh_fr["FR3"] + vh_cdrs["H3"] + vh_fr["FR4"])

    # VL 조립: FR1-CDR-L1-FR2-CDR-L2-FR3-CDR-L3-FR4
    vl = (vl_fr["FR1"] + vl_cdrs["L1"] + vl_fr["FR2"] +
          vl_cdrs["L2"] + vl_fr["FR3"] + vl_cdrs["L3"] + vl_fr["FR4"])

    return vh, vl


# ─── PDB 다운로드 & FAP 구조 준비 ─────────────────────────────────────────────
def download_fap_pdb(pdb_id: str = "1Z68", out_dir: Path = Path("fap_design")):
    """FAP PDB 다운로드"""
    import urllib.request
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = out_dir / f"{pdb_id.lower()}.pdb"
    if not pdb_path.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"[다운로드] {pdb_id} from RCSB...")
        urllib.request.urlretrieve(url, pdb_path)
        print(f"  → {pdb_path}")
    else:
        print(f"  [캐시] {pdb_path}")
    return pdb_path


def prepare_fap_epitope_pdb(pdb_path: Path, out_dir: Path,
                             chain: str = "A",
                             ecd_start: int = 51) -> Path:
    """
    FAP ECD만 추출 (신호펩타이드·TM 제거)
    PDB 1Z68: 체인 A = 모노머 1, 체인 B = 모노머 2
    """
    try:
        from Bio import PDB
    except ImportError:
        print("biopython 필요: pip install biopython")
        return pdb_path

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("FAP", pdb_path)

    # ECD 잔기만 선택 (잔기 번호 >= ecd_start)
    class EcdSelect(PDB.Select):
        def accept_chain(self, c):
            return c.get_id() == chain
        def accept_residue(self, r):
            return r.get_id()[1] >= ecd_start

    io = PDB.PDBIO()
    io.set_structure(structure)
    ecd_path = out_dir / f"fap_ecd_{chain}.pdb"
    io.save(str(ecd_path), EcdSelect())
    print(f"  [FAP ECD] {ecd_path} (잔기 {ecd_start}+ 체인 {chain})")
    return ecd_path


# ─── RFdiffusion 입력 생성 (RTX PC용) ────────────────────────────────────────
def generate_rfdiffusion_input(fap_ecd_pdb: Path, out_dir: Path,
                               hotspots: list = None):
    """
    RFdiffusion contigs 및 hotspot 파일 생성
    실행: RTX 3060 Ti PC (CUDA 필요)
    """
    if hotspots is None:
        hotspots = FAP_TARGET["epitope_hotspots"]

    out_dir.mkdir(parents=True, exist_ok=True)

    # RFdiffusion contig 문자열
    # 형식: A51-390/0 50-150  (FAP ECD + scFv 크기)
    contig = f"A51-390/0 120-130"  # scFv VH ~120 aa

    # hotspot 형식: A308,A310,A312,...
    chain = FAP_TARGET["chain"]
    hotspot_str = ",".join(f"{chain}{r}" for r in hotspots)

    config = {
        "contigmap": {"contigs": [contig]},
        "ppi": {"hotspot_res": hotspots},
        "inference": {
            "input_pdb": str(fap_ecd_pdb),
            "num_designs": 20,
            "output_prefix": str(out_dir / "fap_binder"),
        }
    }

    config_path = out_dir / "rfdiffusion_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # 실행 스크립트
    script_path = out_dir / "run_rfdiffusion.sh"
    with open(script_path, "w") as f:
        f.write(f"""#!/bin/bash
# RFdiffusion FAP binder 설계 (RTX PC 실행)
# GPU: CUDA 11.8+, VRAM 8GB+

FAP_PDB="{fap_ecd_pdb}"
HOTSPOTS="{hotspot_str}"
OUT_DIR="{out_dir}"

python /path/to/RFdiffusion/scripts/run_inference.py \\
    inference.input_pdb=$FAP_PDB \\
    'contigmap.contigs=[A51-390/0 120-130]' \\
    "ppi.hotspot_res=[$HOTSPOTS]" \\
    inference.num_designs=20 \\
    inference.output_prefix=$OUT_DIR/fap_binder

echo "RFdiffusion 완료: $OUT_DIR/fap_binder_*.pdb"
echo "다음 단계: ProteinMPNN로 서열 설계"
""")
    os.chmod(script_path, 0o755)

    print(f"  [RFdiffusion] 설정: {config_path}")
    print(f"  [RFdiffusion] 실행 스크립트: {script_path}")
    print(f"  [핫스팟] {hotspot_str}")
    return config_path


# ─── ProteinMPNN 서열 설계 (M5 Pro / RTX 공통) ───────────────────────────────
def generate_proteinmpnn_script(backbone_pdb_dir: Path, out_dir: Path,
                                 mpnn_dir: str = "~/tools/ProteinMPNN"):
    """RFdiffusion 백본에서 서열 설계 (ProteinMPNN)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / "run_proteinmpnn.sh"
    with open(script_path, "w") as f:
        f.write(f"""#!/bin/bash
# ProteinMPNN 서열 설계 (M5 Pro MPS 또는 RTX CUDA)

MPNN_DIR="{mpnn_dir}"
BACKBONE_DIR="{backbone_pdb_dir}"
OUT_DIR="{out_dir}/mpnn_seqs"
mkdir -p $OUT_DIR

# 각 RFdiffusion 백본에 대해 서열 설계 (각 3개 서열)
for pdb in $BACKBONE_DIR/fap_binder_*.pdb; do
    name=$(basename $pdb .pdb)
    python $MPNN_DIR/protein_mpnn_run.py \\
        --pdb_path $pdb \\
        --out_folder $OUT_DIR/$name \\
        --num_seq_per_target 3 \\
        --sampling_temp 0.1 \\
        --score_only 0 \\
        --seed 42
done

echo "ProteinMPNN 완료: $OUT_DIR"
echo "다음 단계: IgFold 구조 예측 → ColabFold 복합체 검증"
""")
    os.chmod(script_path, 0o755)
    print(f"  [ProteinMPNN] 스크립트: {script_path}")


# ─── scFv FASTA 출력 ──────────────────────────────────────────────────────────
def write_scfv_fasta(vh: str, vl: str, name: str, out_path: Path,
                     orientation: str = "VH-VL"):
    """scFv FASTA 파일 작성"""
    scfv = build_scfv(vh, vl, orientation)
    with open(out_path, "w") as f:
        f.write(f">{name}_scFv_{orientation}\n{scfv}\n")
        f.write(f">{name}_VH\n{vh}\n")
        f.write(f">{name}_VL\n{vl}\n")
    print(f"  [FASTA] {out_path}")
    return scfv


# ─── 현황 요약 출력 ───────────────────────────────────────────────────────────
def print_design_summary():
    vh_fr, vl_fr, vh_cdrs, vl_cdrs = get_trastuzumab_frameworks()

    print("=" * 60)
    print("  FAP scFv 설계 계획 (CAR-T)")
    print("=" * 60)
    print()
    print("[ 표적 ]")
    print(f"  항원     : FAP (FAPα/Seprase, {FAP_TARGET['uniprot']})")
    print(f"  PDB      : {FAP_TARGET['pdb']} (β-프로펠러 호모다이머)")
    print(f"  에피토프 : β-프로펠러 Blade 6-7")
    print(f"  핫스팟   : {FAP_TARGET['epitope_hotspots']}")
    print()
    print("[ 프레임워크: 트라스투주맙 ]")
    print(f"  VH FR1: {vh_fr.get('FR1', '')}")
    print(f"  VH FR2: {vh_fr.get('FR2', '')}")
    print(f"  VH FR3: {vh_fr.get('FR3', '')}")
    print(f"  VH FR4: {vh_fr.get('FR4', '')}")
    print()
    print(f"  VL FR1: {vl_fr.get('FR1', '')}")
    print(f"  VL FR2: {vl_fr.get('FR2', '')}")
    print(f"  VL FR3: {vl_fr.get('FR3', '')}")
    print(f"  VL FR4: {vl_fr.get('FR4', '')}")
    print()
    print("[ 트라스투주맙 원래 CDR (교체 예정) ]")
    print(f"  CDR-H1 ({VH_CDR_CHOTHIA['H1']}): {vh_cdrs.get('H1', '')}")
    print(f"  CDR-H2 ({VH_CDR_CHOTHIA['H2']}): {vh_cdrs.get('H2', '')}")
    print(f"  CDR-H3 ({VH_CDR_CHOTHIA['H3']}): {vh_cdrs.get('H3', '')}")
    print(f"  CDR-L1 ({VL_CDR_CHOTHIA['L1']}): {vl_cdrs.get('L1', '')}")
    print(f"  CDR-L2 ({VL_CDR_CHOTHIA['L2']}): {vl_cdrs.get('L2', '')}")
    print(f"  CDR-L3 ({VL_CDR_CHOTHIA['L3']}): {vl_cdrs.get('L3', '')}")
    print()
    print("[ 파이프라인 ]")
    print("  1. FAP PDB 1Z68 다운로드 → ECD 추출")
    print("  2. RFdiffusion (RTX)     → VH/VL 백본 20개 생성")
    print("  3. ProteinMPNN (M5/RTX)  → 각 백본 × 3 서열")
    print("  4. 트라스투주맙 FR 이식  → 60개 후보 scFv")
    print("  5. IgFold (M5)           → 구조 예측")
    print("  6. ColabFold/Boltz-2(M5) → FAP 복합체 검증")
    print("  7. ESM-2 PLL (M5)        → 서열 안정성 스코어")
    print("  8. OpenMM ff19SB (M5)    → 100 ns MD 검증")
    print()
    print("[ scFv 형식 ]")
    scfv_demo = build_scfv("VH...", "VL...", "VH-VL", SCFV_LINKER_G4S3)
    print(f"  VH-(G4S)3-VL")
    print(f"  링커: {SCFV_LINKER_G4S3} (15 aa)")
    print()
    print("[ CAR 구조 ]")
    print("  scFv-CD8α힌지-CD8α TM-4-1BB-CD3ζ (2세대 CAR)")
    print("=" * 60)


# ─── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FAP scFv 설계 (CAR-T)")
    parser.add_argument("--out_dir", default="fap_design",
                        help="출력 디렉터리")
    parser.add_argument("--download_pdb", action="store_true",
                        help="FAP PDB 1Z68 다운로드")
    parser.add_argument("--gen_rfdiffusion", action="store_true",
                        help="RFdiffusion 입력 파일 생성 (RTX에서 실행)")
    parser.add_argument("--gen_mpnn", action="store_true",
                        help="ProteinMPNN 스크립트 생성")
    parser.add_argument("--graft_cdrs", nargs=6, metavar=("H1","H2","H3","L1","L2","L3"),
                        help="CDR 서열 이식: H1 H2 H3 L1 L2 L3")
    parser.add_argument("--summary", action="store_true", default=True,
                        help="설계 요약 출력")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary:
        print_design_summary()

    if args.download_pdb:
        pdb_path = download_fap_pdb("1Z68", out_dir)
        ecd_path = prepare_fap_epitope_pdb(pdb_path, out_dir)

    if args.gen_rfdiffusion:
        ecd_path = out_dir / "fap_ecd_A.pdb"
        if not ecd_path.exists():
            pdb_path = download_fap_pdb("1Z68", out_dir)
            ecd_path = prepare_fap_epitope_pdb(pdb_path, out_dir)
        generate_rfdiffusion_input(ecd_path, out_dir / "rfdiffusion")
        generate_proteinmpnn_script(out_dir / "rfdiffusion", out_dir / "mpnn")
        print()
        print("[다음 단계]")
        print(f"  RTX PC: bash {out_dir}/rfdiffusion/run_rfdiffusion.sh")
        print(f"  RTX PC: bash {out_dir}/mpnn/run_proteinmpnn.sh")

    if args.graft_cdrs:
        h1, h2, h3, l1, l2, l3 = args.graft_cdrs
        vh_cdrs = {"H1": h1, "H2": h2, "H3": h3}
        vl_cdrs = {"L1": l1, "L2": l2, "L3": l3}
        vh, vl = graft_cdrs_into_trastuzumab(vh_cdrs, vl_cdrs)
        fasta_path = out_dir / "fap_scfv_candidate.fasta"
        scfv = write_scfv_fasta(vh, vl, "FAP_scFv", fasta_path)
        print(f"\n[scFv 서열] {len(scfv)} aa")
        print(f"  VH: {vh}")
        print(f"  VL: {vl}")
        print(f"\n[다음 단계]")
        print(f"  python src/run_esm2_score.py --vh '{vh}' --vl '{vl}'")
        print(f"  # IgFold 구조 예측 후 ColabFold로 FAP 복합체 검증")

    # 설계 정보 저장
    design_info = {
        "target": FAP_TARGET,
        "framework": "trastuzumab",
        "framework_vh": TRASTUZUMAB_VH,
        "framework_vl": TRASTUZUMAB_VL,
        "linker": SCFV_LINKER_G4S3,
        "car_format": "scFv-CD8a_hinge-CD8a_TM-4-1BB-CD3z",
        "cdr_boundaries_chothia": {
            "VH": VH_CDR_CHOTHIA,
            "VL": VL_CDR_CHOTHIA,
        }
    }
    info_path = out_dir / "design_info.json"
    with open(info_path, "w") as f:
        json.dump(design_info, f, indent=2, ensure_ascii=False)
    print(f"\n[설계 정보] {info_path}")


if __name__ == "__main__":
    main()
