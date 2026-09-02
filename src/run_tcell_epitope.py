#!/usr/bin/env python3
"""
T-cell epitope risk scoring for FAP scFv Top 5 candidates.
Uses MHCflurry (MHC-I) + heuristic MHC-II window scan.
Input:  fap_design/candidates/top5_final.json
Output: fap_design/tcell/tcell_epitope_results.json
"""

import argparse
import json
import os
import sys

# ── MHC-I alleles (common HLA supertypes) ──────────────────────────────────
MHC1_ALLELES = [
    "HLA-A*02:01",
    "HLA-A*03:01",
    "HLA-A*24:02",
    "HLA-B*07:02",
    "HLA-B*35:01",
]

# ── MHC-II binding heuristic: hydrophobic / aromatic enrichment ─────────────
HYDROPHOBIC = set("VILMFYW")
MHC2_WINDOW = 15  # 15-mer sliding window
MHC2_THRESHOLD = 5  # ≥5 hydrophobic in 15-mer → potential binder


def mhc1_score(runner, seq, alleles, peptide_lengths=(9, 10)):
    """Return list of high-affinity 9/10-mers (IC50 < 500 nM)."""
    hits = []
    for plen in peptide_lengths:
        for i in range(len(seq) - plen + 1):
            pep = seq[i:i + plen]
            try:
                result = runner.predict(
                    peptides=[pep],
                    alleles=alleles,
                )
                for _, row in result.iterrows():
                    if row["mhcflurry_affinity"] < 500:
                        hits.append({
                            "peptide": pep,
                            "allele": row["allele"],
                            "ic50_nM": round(float(row["mhcflurry_affinity"]), 1),
                            "pos": i,
                        })
            except Exception:
                pass
    return hits


def mhc2_heuristic(seq, window=MHC2_WINDOW, threshold=MHC2_THRESHOLD):
    """Simple hydrophobic-content heuristic for MHC-II risk."""
    hits = []
    for i in range(len(seq) - window + 1):
        pep = seq[i:i + window]
        n_hydro = sum(1 for aa in pep if aa in HYDROPHOBIC)
        if n_hydro >= threshold:
            hits.append({
                "peptide": pep,
                "pos": i,
                "hydrophobic_count": n_hydro,
                "risk": "HIGH" if n_hydro >= 7 else "MOD",
            })
    return hits


def score_candidate(cand, mhc1_runner=None):
    cid = cand["id"]
    vh = cand["vh"]
    vl = cand["vl"]
    cdrs = cand.get("cdrs", {})

    result = {"id": cid, "cdrs": cdrs}

    # MHC-I
    if mhc1_runner is not None:
        mhc1_hits = mhc1_score(mhc1_runner, vh + vl, MHC1_ALLELES)
        result["mhc1_hits"] = mhc1_hits
        result["mhc1_n_hits"] = len(mhc1_hits)
        result["mhc1_risk"] = "HIGH" if len(mhc1_hits) >= 3 else ("MOD" if mhc1_hits else "LOW")
    else:
        result["mhc1_hits"] = []
        result["mhc1_n_hits"] = None
        result["mhc1_risk"] = "N/A (MHCflurry not available)"

    # MHC-II heuristic (VH + VL)
    mhc2_hits_vh = mhc2_heuristic(vh)
    mhc2_hits_vl = mhc2_heuristic(vl)
    mhc2_hits = mhc2_hits_vh + mhc2_hits_vl
    result["mhc2_heuristic_hits"] = mhc2_hits
    result["mhc2_n_hits"] = len(mhc2_hits)
    result["mhc2_risk"] = "HIGH" if len(mhc2_hits) >= 3 else ("MOD" if mhc2_hits else "LOW")

    # CDR-specific MHC-II
    cdr_mhc2 = {}
    for key, seq in cdrs.items():
        if len(seq) >= MHC2_WINDOW:
            cdr_mhc2[key] = mhc2_heuristic(seq, window=min(len(seq), MHC2_WINDOW))
        else:
            # pad-extend with framework flanks
            cdr_mhc2[key] = []
    result["cdr_mhc2"] = cdr_mhc2

    # Overall risk
    risks = [result["mhc2_risk"]]
    if result["mhc1_risk"] not in ("N/A (MHCflurry not available)",):
        risks.append(result["mhc1_risk"])
    risk_rank = {"HIGH": 2, "MOD": 1, "LOW": 0}
    overall = max(risks, key=lambda r: risk_rank.get(r, -1))
    result["overall_risk"] = overall

    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="fap_design/candidates/top5_final.json")
    p.add_argument("--out_dir", default="fap_design/tcell")
    p.add_argument("--fasta", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.json) as f:
        top5 = json.load(f)

    # Try to load MHCflurry
    mhc1_runner = None
    try:
        from mhcflurry import Class1PresentationPredictor
        print("[MHCflurry] 로딩 중...")
        mhc1_runner = Class1PresentationPredictor.load()
        print("[MHCflurry] 모델 로드 완료.\n")
    except ImportError:
        print("[MHCflurry] 미설치 — MHC-I 스킵, MHC-II 휴리스틱만 실행.\n")
    except Exception as e:
        print(f"[MHCflurry] 오류: {e} — MHC-I 스킵.\n")

    results = []
    for i, cand in enumerate(top5, 1):
        cid = cand["id"]
        print(f"[{i}/{len(top5)}] {cid} 스코어링...")
        r = score_candidate(cand, mhc1_runner)
        results.append(r)

        mhc1_s = f"MHC-I {r['mhc1_risk']} ({r['mhc1_n_hits']} hits)" if r["mhc1_n_hits"] is not None else "MHC-I N/A"
        print(f"  {mhc1_s}  |  MHC-II(heur) {r['mhc2_risk']} ({r['mhc2_n_hits']} hits)  |  Overall: {r['overall_risk']}")

    # Save results
    out_path = os.path.join(args.out_dir, "tcell_epitope_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] 결과 저장: {out_path}")
    print("\n=== T세포 에피토프 요약 ===")
    print(f"{'ID':<18} {'MHC-I':>10} {'MHC-II':>7} {'Overall':>8}")
    print("-" * 48)
    for r in results:
        mhc1_s = r["mhc1_risk"] if r["mhc1_n_hits"] is not None else "N/A"
        print(f"{r['id']:<18} {mhc1_s:>10} {r['mhc2_risk']:>7} {r['overall_risk']:>8}")

    print("\n다음: git add fap_design/tcell/ && git commit -m 'data: T세포 에피토프' && git push")


if __name__ == "__main__":
    main()
