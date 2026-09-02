#!/usr/bin/env python3
"""
ColabFold 복합체 결과 분석
- ipTM / pTM 추출
- Blade 6-7 접촉 분석 (FAP 308-361, CDR H1/H2/H3/L1/L2/L3)
- 상위 모델 PDB 저장

Input:  fap_design/colabfold/results/<id>_scores*.json  (ColabFold 출력)
Output: fap_design/colabfold/complex_summary.json
        fap_design/colabfold/contact_analysis.tsv
"""

import argparse
import glob
import json
import os
import sys

# Blade 6-7 FAP 잔기 번호 (1Z68 chain A 기준)
BLADE67_RANGE = (308, 361)
# 핵심 에피토프 잔기
KEY_RESIDUES = {311: "E311", 313: "D313", 356: "R356", 358: "F358", 360: "K360"}

# scFv CDR 잔기 구간 (Chothia, scFv 내 절대위치)
# VH: 1-120, VL: linker 20aa + 107aa = 121-246
# Linker (G4S)x4 = 16aa (real: GGGGSGGGGSGGGGSGGGGS)
VH_LEN = 120  # approximate
LINKER_LEN = 20  # (GGGGS)x4 = 20aa
VL_START = VH_LEN + LINKER_LEN  # chain B residue offset

CDR_RANGES_VH = {
    "H1": (26, 32),    # Chothia
    "H2": (52, 56),
    "H3": (97, 108),   # variable end
}
CDR_RANGES_VL = {
    "L1": (24, 34),
    "L2": (50, 56),
    "L3": (89, 97),
}

CONTACT_CUTOFF_ANG = 4.5  # Å


def parse_scores_json(scores_path: str) -> dict:
    """ColabFold scores JSON → iptm, ptm, plddt_mean."""
    with open(scores_path) as f:
        d = json.load(f)
    return {
        "iptm": d.get("iptm", None),
        "ptm": d.get("ptm", None),
        "plddt_mean": round(sum(d.get("plddt", [0])) / max(1, len(d.get("plddt", [1]))), 3),
        "ranking_confidence": d.get("ranking_confidence", None),
    }


def find_best_model(result_dir: str, cand_id: str) -> tuple:
    """Find best model by ranking_confidence from ColabFold output."""
    pattern = os.path.join(result_dir, f"*{cand_id}*scores*.json")
    score_files = sorted(glob.glob(pattern))
    if not score_files:
        # try without cand_id filter
        pattern = os.path.join(result_dir, "*scores*.json")
        score_files = sorted(glob.glob(pattern))

    best_score = None
    best_pdb = None
    best_meta = {}

    for sf in score_files:
        meta = parse_scores_json(sf)
        rc = meta.get("ranking_confidence") or meta.get("iptm") or 0
        if best_score is None or rc > best_score:
            best_score = rc
            best_meta = meta
            # Corresponding PDB
            pdb_path = sf.replace("_scores_rank_", "_unrelaxed_rank_").replace(".json", ".pdb")
            if not os.path.exists(pdb_path):
                pdb_path = sf.replace("scores", "relaxed").replace(".json", ".pdb")
            best_pdb = pdb_path if os.path.exists(pdb_path) else None

    return best_pdb, best_meta


def contact_analysis_pdb(pdb_path: str, fap_len: int) -> dict:
    """
    Simple Cα contact analysis between FAP chain (chain A) and scFv (chain B).
    Returns: blade67_contacts, key_residue_contacts
    """
    if not pdb_path or not os.path.exists(pdb_path):
        return {"error": "PDB not found", "blade67_contacts": [], "key_contacts": {}}

    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy not available", "blade67_contacts": [], "key_contacts": {}}

    # Parse Cα atoms from PDB
    chainA_ca = {}  # resnum → (x, y, z)
    chainB_ca = {}

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain = line[21]
            resnum = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            if chain == "A":
                chainA_ca[resnum] = np.array([x, y, z])
            elif chain == "B":
                chainB_ca[resnum] = np.array([x, y, z])

    # Blade 6-7 contacts: FAP 308-361 ↔ scFv any
    blade67_contacts = []
    for fap_res in range(BLADE67_RANGE[0], BLADE67_RANGE[1] + 1):
        if fap_res not in chainA_ca:
            continue
        fap_coord = chainA_ca[fap_res]
        for scfv_res, scfv_coord in chainB_ca.items():
            dist = float(np.linalg.norm(fap_coord - scfv_coord))
            if dist <= CONTACT_CUTOFF_ANG:
                blade67_contacts.append({
                    "fap_res": fap_res,
                    "scfv_res": scfv_res,
                    "dist_ang": round(dist, 2),
                    "fap_label": KEY_RESIDUES.get(fap_res, f"FAP{fap_res}"),
                })

    # Key residue contacts
    key_contacts = {}
    for fap_res, label in KEY_RESIDUES.items():
        if fap_res not in chainA_ca:
            key_contacts[label] = None
            continue
        fap_coord = chainA_ca[fap_res]
        contacts = []
        for scfv_res, scfv_coord in chainB_ca.items():
            dist = float(np.linalg.norm(fap_coord - scfv_coord))
            if dist <= CONTACT_CUTOFF_ANG:
                contacts.append({"scfv_res": scfv_res, "dist_ang": round(dist, 2)})
        key_contacts[label] = sorted(contacts, key=lambda x: x["dist_ang"])[:3]

    return {
        "blade67_contacts": blade67_contacts,
        "n_blade67_contacts": len(blade67_contacts),
        "key_contacts": key_contacts,
        "blade67_contacted": len(blade67_contacts) > 0,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", default="fap_design/colabfold/results")
    p.add_argument("--out_dir", default="fap_design/colabfold")
    p.add_argument("--fap_len", type=int, default=367,
                   help="FAP ECD 서열 길이 (chain A)")
    p.add_argument("--candidates", nargs="+",
                   default=["FAP-scFv-12534", "FAP-scFv-13034", "FAP-scFv-6446",
                            "FAP-scFv-1536", "FAP-scFv-4766"])
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.result_dir):
        print(f"[ERROR] ColabFold 결과 디렉토리 없음: {args.result_dir}")
        print("먼저 bash fap_design/colabfold/run_colabfold.sh 실행")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print("=== ColabFold 복합체 분석 ===\n")
    results = []

    for cand_id in args.candidates:
        print(f"[{cand_id}] 분석 중...")
        best_pdb, meta = find_best_model(args.result_dir, cand_id)
        contacts = contact_analysis_pdb(best_pdb, args.fap_len)

        result = {
            "id": cand_id,
            "best_pdb": best_pdb,
            **meta,
            "blade67_contacts": contacts.get("n_blade67_contacts", 0),
            "blade67_contacted": contacts.get("blade67_contacted", False),
            "key_contacts": contacts.get("key_contacts", {}),
            "pass_iptm": (meta.get("iptm") or 0) >= 0.5,
        }
        results.append(result)

        iptm_s = f"{meta['iptm']:.3f}" if meta.get("iptm") else "N/A"
        ptm_s = f"{meta['ptm']:.3f}" if meta.get("ptm") else "N/A"
        n_contacts = contacts.get("n_blade67_contacts", 0)
        print(f"  ipTM={iptm_s}  pTM={ptm_s}  Blade6-7_contacts={n_contacts}")
        for label, ctcts in contacts.get("key_contacts", {}).items():
            if ctcts:
                best = ctcts[0]
                print(f"    {label} → scFv_res{best['scfv_res']} {best['dist_ang']}Å")
        print()

    # Save summary
    summary_path = os.path.join(args.out_dir, "complex_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[완료] 요약: {summary_path}")

    # TSV output
    tsv_path = os.path.join(args.out_dir, "contact_analysis.tsv")
    with open(tsv_path, "w") as f:
        f.write("ID\tipTM\tpTM\tpLDDT\tBlade6-7_contacts\tPass_ipTM\n")
        for r in results:
            iptm = f"{r['iptm']:.3f}" if r.get("iptm") else "N/A"
            ptm = f"{r['ptm']:.3f}" if r.get("ptm") else "N/A"
            plddt = f"{r['plddt_mean']:.1f}" if r.get("plddt_mean") else "N/A"
            f.write(f"{r['id']}\t{iptm}\t{ptm}\t{plddt}\t{r['blade67_contacts']}\t{r['pass_iptm']}\n")
    print(f"[완료] TSV: {tsv_path}")

    # Rank by ipTM
    ranked = sorted(results, key=lambda x: x.get("iptm") or 0, reverse=True)
    print("\n=== ipTM 순위 ===")
    print(f"{'ID':<22} {'ipTM':>6} {'pTM':>6} {'Blade6-7':>9} {'Pass':>5}")
    print("-" * 55)
    for r in ranked:
        iptm_s = f"{r['iptm']:.3f}" if r.get("iptm") else "  N/A"
        ptm_s = f"{r['ptm']:.3f}" if r.get("ptm") else "  N/A"
        pass_s = "✅" if r["pass_iptm"] else "❌"
        print(f"{r['id']:<22} {iptm_s:>6} {ptm_s:>6} {r['blade67_contacts']:>9} {pass_s:>5}")


if __name__ == "__main__":
    main()
