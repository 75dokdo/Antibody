#!/usr/bin/env python3
"""
IgFold structure prediction for FAP scFv Top 5 candidates.
Input: fap_design/candidates/top5_final.json (VH/VL sequences)
Output: fap_design/structures/<id>.pdb + <id>_rmsd.txt
"""

import argparse
import json
import os
import sys

def parse_args():
    p = argparse.ArgumentParser(description="IgFold structure prediction for FAP scFv Top 5")
    p.add_argument("--json", default="fap_design/candidates/top5_final.json",
                   help="top5_final.json path")
    p.add_argument("--out_dir", default="fap_design/structures",
                   help="Output directory for PDB files")
    p.add_argument("--fasta", default=None,
                   help="(unused, kept for CLI compat)")
    return p.parse_args()


def fold_candidate(runner, cand_id, vh, vl, out_path):
    """Fold one scFv and save PDB."""
    sequences = {"H": vh, "L": vl}
    print(f"  [IgFold] Folding {cand_id} (VH={len(vh)}aa, VL={len(vl)}aa)...")
    try:
        pred = runner.fold(sequences=sequences)
        pred.save_pdb(out_path)
        # pLDDT mean from output
        try:
            plddt = float(pred.plddt.mean())
        except Exception:
            plddt = None
        return plddt
    except Exception as e:
        print(f"  [IgFold] ERROR: {e}", file=sys.stderr)
        return None


def main():
    args = parse_args()

    # Import after pip install
    try:
        from igfold import IgFoldRunner
    except ImportError:
        print("[ERROR] igfold not installed. Run: pip3 install igfold", file=sys.stderr)
        sys.exit(1)

    # Load top5
    with open(args.json) as f:
        top5 = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[IgFold] 로딩 중...")
    runner = IgFoldRunner()
    print(f"[IgFold] 모델 로드 완료. {len(top5)}개 후보 예측 시작.\n")

    results = []
    for i, cand in enumerate(top5, 1):
        cid = cand["id"]
        vh = cand["vh"]
        vl = cand["vl"]
        out_pdb = os.path.join(args.out_dir, f"{cid}.pdb")

        print(f"[{i}/{len(top5)}] {cid}")
        plddt = fold_candidate(runner, cid, vh, vl, out_pdb)

        if plddt is not None:
            print(f"  → 저장: {out_pdb}  pLDDT={plddt:.2f}")
        else:
            print(f"  → 실패: {out_pdb}")

        results.append({
            "id": cid,
            "pdb": out_pdb,
            "plddt": round(plddt, 3) if plddt else None,
            "h3": cand.get("cdrs", {}).get("H3", ""),
        })

    # Save summary
    summary_path = os.path.join(args.out_dir, "igfold_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[IgFold] 완료. 결과:")
    print(f"{'ID':<18} {'pLDDT':>7}  H3")
    print("-" * 55)
    for r in results:
        plddt_s = f"{r['plddt']:.2f}" if r["plddt"] else "FAIL"
        print(f"{r['id']:<18} {plddt_s:>7}  {r['h3']}")

    print(f"\n요약 저장: {summary_path}")
    print("다음: git add fap_design/structures/ && git commit -m 'data: IgFold PDB' && git push")


if __name__ == "__main__":
    main()
