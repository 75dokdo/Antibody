"""Screen candidate GPC3 epitope regions against known / patented epitopes.

Usage:
    python3 src/epitope_fto_scan.py
    python3 src/epitope_fto_scan.py --json results/fto_scan.json

For every candidate region file the scan reports:

  * how many of its residues collide with each published or patent-claimed
    epitope (in UniProt numbering, after correcting the +24 offset),
  * how much of the region is actually solvent-exposed in the intact
    ectodomain -- buried residues cannot form an antibody epitope,
  * the exposed residues that collide with nothing, which are the patches a
    new campaign can realistically claim.

This is a triage aid, not a legal opinion.  See the DISCLAIMER in the
patent database.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpc3_lib import (
    PATENT_DB,
    STRUCTURE_DIR,
    Residue,
    epitope_uniprot_positions,
    load_residues,
    sequence_of,
)

# Evidence level -> how much weight an overlap carries.  A collision with a
# range we only know at "N-lobe" granularity should not outrank a collision
# with a peptide that a granted claim spells out letter by letter.
EVIDENCE_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.35}

#: Entries whose published epitope is only resolved to a whole lobe/domain.
#: They span hundreds of residues, so letting them mark surface as "claimed"
#: would black out an entire lobe on the strength of a single mAb that in
#: reality contacts one patch of it.  They are reported as advisories instead.
DOMAIN_LEVEL_IDS = {"HN3", "HN3_competing_cluster"}


def contiguous_runs(positions: list[int]) -> list[tuple[int, int]]:
    """Collapse a sorted list of ints into (start, end) runs."""
    runs: list[tuple[int, int]] = []
    for pos in sorted(positions):
        if runs and pos == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], pos)
        else:
            runs.append((pos, pos))
    return runs


def format_runs(positions: list[int], limit: int = 8) -> str:
    runs = contiguous_runs(positions)
    parts = [f"{a}-{b}" if a != b else str(a) for a, b in runs[:limit]]
    if len(runs) > limit:
        parts.append(f"(+{len(runs) - limit} more)")
    return ", ".join(parts) if parts else "-"


def scan_region(
    name: str,
    residues: list[Residue],
    sasa_by_uniprot: dict[int, Residue],
    epitopes: list[dict],
    hotspots: list[dict],
) -> dict:
    """Collide one candidate region against the whole reference database."""
    # Use the burial computed on the intact ectodomain, not on the fragment.
    for res in residues:
        ref = sasa_by_uniprot.get(res.uniprot)
        if ref is not None:
            res.sasa, res.rsa = ref.sasa, ref.rsa

    region_positions = {r.uniprot for r in residues}
    exposed_positions = {r.uniprot for r in residues if r.exposed}

    collisions = []
    advisories = []
    claimed_positions: set[int] = set()
    for epitope in epitopes:
        ep_positions = epitope_uniprot_positions(epitope)
        overlap = region_positions & ep_positions
        if not overlap:
            continue

        exposed_overlap = overlap & exposed_positions
        domain_level = epitope["id"] in DOMAIN_LEVEL_IDS
        if not domain_level:
            claimed_positions |= exposed_overlap

        core = epitope.get("core_uniprot_residues")
        core_overlap = set()
        if core:
            core_overlap = overlap & set(range(core["start"], core["end"] + 1))

        weight = EVIDENCE_WEIGHT.get(epitope.get("evidence", "medium"), 0.7)
        # Score on the exposed fraction: a collision buried under the surface
        # is not a collision an antibody can ever experience.
        score = weight * len(exposed_overlap) / max(len(exposed_positions), 1)
        if core_overlap:
            score = max(score, weight)  # any core-epitope contact is decisive

        record = {
            "epitope_id": epitope["id"],
            "antibody": epitope["antibody"],
            "owner": epitope.get("owner", ""),
            "patents": epitope.get("patents", []),
            "evidence": epitope.get("evidence"),
            "epitope_type": epitope.get("epitope_type"),
            "overlap_residues": len(overlap),
            "exposed_overlap_residues": len(exposed_overlap),
            "core_overlap_residues": sorted(core_overlap),
            "overlap_uniprot": format_runs(sorted(overlap)),
            "score": round(score, 3),
            "domain_level": domain_level,
        }
        (advisories if domain_level else collisions).append(record)

    collisions.sort(key=lambda c: c["score"], reverse=True)
    advisories.sort(key=lambda c: c["score"], reverse=True)

    hotspot_hits = []
    for spot in hotspots:
        span = spot["uniprot_residues"]
        overlap = region_positions & set(range(span["start"], span["end"] + 1))
        if overlap:
            hotspot_hits.append(
                {
                    "id": spot["id"],
                    "type": spot["type"],
                    "overlap_uniprot": format_runs(sorted(overlap)),
                    "key_residue": spot.get("key_residue"),
                    "notes": spot["notes"],
                }
            )

    free_exposed = sorted(exposed_positions - claimed_positions)
    top_score = collisions[0]["score"] if collisions else 0.0
    if top_score >= 0.7:
        verdict = "HIGH RISK"
    elif top_score >= 0.3:
        verdict = "MEDIUM RISK"
    elif top_score > 0:
        verdict = "LOW RISK"
    else:
        verdict = "CLEAR"

    return {
        "region": name,
        "mature_range": [residues[0].mature, residues[-1].mature],
        "uniprot_range": [residues[0].uniprot, residues[-1].uniprot],
        "n_residues": len(residues),
        "n_exposed": len(exposed_positions),
        "mean_plddt": round(sum(r.plddt for r in residues) / len(residues), 1),
        "sequence": sequence_of(residues),
        "verdict": verdict,
        "risk_score": round(top_score, 3),
        "collisions": collisions,
        "advisories": advisories,
        "functional_hotspots": hotspot_hits,
        "free_exposed_uniprot": format_runs(free_exposed, limit=12),
        "n_free_exposed": len(free_exposed),
        "free_exposed_fraction": round(
            len(free_exposed) / max(len(exposed_positions), 1), 3
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write full results as JSON")
    parser.add_argument(
        "--regions",
        nargs="*",
        help="specific region PDB files (default: every GPC3_region*.pdb)",
    )
    args = parser.parse_args()

    db = json.loads(PATENT_DB.read_text())
    epitopes = db["epitopes"]
    hotspots = db["functional_hotspots"]

    # Burial has to come from the intact ectodomain.
    full = load_residues(STRUCTURE_DIR / "GPC3_ectodomain_full.pdb", with_sasa=True)
    sasa_by_uniprot = {r.uniprot: r for r in full}

    paths = (
        [Path(p) for p in args.regions]
        if args.regions
        else sorted(STRUCTURE_DIR.glob("GPC3_region*.pdb"))
    )

    print("=" * 78)
    print("GPC3 candidate-epitope screen against published / patented epitopes")
    print("=" * 78)
    print(f"Reference database : {len(epitopes)} epitopes, {len(hotspots)} hotspots")
    print(f"Numbering          : UniProt = mature + {db['_meta']['numbering_offset_uniprot_minus_mature']}")
    print(f"Ectodomain model   : {len(full)} residues, "
          f"{sum(1 for r in full if r.exposed)} solvent-exposed")
    print()

    results = []
    for path in paths:
        residues = load_residues(path)
        report = scan_region(path.stem, residues, sasa_by_uniprot, epitopes, hotspots)
        results.append(report)

        print("-" * 78)
        print(f"{report['region']}   mature {report['mature_range'][0]}-{report['mature_range'][1]}"
              f"   |   UniProt {report['uniprot_range'][0]}-{report['uniprot_range'][1]}")
        print("-" * 78)
        print(f"  residues {report['n_residues']}, exposed {report['n_exposed']}, "
              f"mean pLDDT {report['mean_plddt']}")
        print(f"  VERDICT: {report['verdict']}  (score {report['risk_score']})")

        if report["collisions"]:
            print("  collisions:")
            for c in report["collisions"]:
                core = (f", CORE EPITOPE HIT {c['core_overlap_residues']}"
                        if c["core_overlap_residues"] else "")
                print(f"    - {c['epitope_id']:<22} {c['antibody']}")
                print(f"      {c['overlap_residues']} res overlap "
                      f"({c['exposed_overlap_residues']} exposed) at {c['overlap_uniprot']}{core}")
                print(f"      evidence={c['evidence']}  score={c['score']}  "
                      f"patents={', '.join(c['patents'][:2])}")
        else:
            print("  collisions: none (no residue-level claim touches this region)")

        if report["advisories"]:
            print("  advisories (domain-level epitopes, not residue-resolved):")
            for a in report["advisories"]:
                print(f"    ~ {a['epitope_id']:<22} {a['antibody']}")
                print(f"      region lies inside the claimed {a['overlap_uniprot']} span; "
                      f"real contact patch is narrower and unpublished")

        if report["functional_hotspots"]:
            print("  functional hotspots in this region:")
            for h in report["functional_hotspots"]:
                print(f"    ! {h['id']} ({h['type']}) at {h['overlap_uniprot']}"
                      f"  key={h['key_residue']}")

        print(f"  unclaimed exposed surface: {report['n_free_exposed']}/{report['n_exposed']} "
              f"residues ({report['free_exposed_fraction']:.0%})")
        print(f"    {report['free_exposed_uniprot']}")
        print()

    print("=" * 78)
    print("RANKING (most patent-free first)")
    print("=" * 78)
    ranked = sorted(results, key=lambda r: (r["risk_score"], -r["free_exposed_fraction"]))
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. {r['region']:<22} {r['verdict']:<12} "
              f"risk={r['risk_score']:<6} free surface={r['free_exposed_fraction']:.0%}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"database": db["_meta"], "regions": results}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
