"""Cluster exposed residues into candidate epitope patches.

The region scan works residue by residue, which answers "is this residue
free" but not the question that actually decides a campaign: is there a
CONTIGUOUS piece of surface big enough for an antibody to sit on, and is
that piece free.

An antibody paratope buries roughly 600-900 A^2 of antigen surface, drawn
from a compact patch of 15-25 residues.  Scattered free residues that never
touch each other cannot be targeted no matter how unencumbered they are.

Two things this does that a sequence-window view cannot:

  * patches are built in 3D, so a patch may recruit residues from sequence
    segments far away -- which is what a real conformational epitope does;
  * every patch is re-checked against the patent database over its FULL
    3D membership, so a patch nucleated in a clean region but leaning into
    a claimed neighbour is caught.

Usage:
    python3 src/surface_patches.py --region GPC3_regionB
    python3 src/surface_patches.py --region GPC3_regionB --json results/patches_B.json
    python3 src/surface_patches.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

from gpc3_lib import (
    PATENT_DB,
    STRUCTURE_DIR,
    epitope_uniprot_positions,
    load_residues,
)

#: Radius of a candidate epitope patch, measured from the patch centre.
#: A patch is every exposed residue whose centroid falls inside this sphere.
#:
#: This replaces connected-component clustering, which percolates: with a
#: contact cutoff every surface residue is transitively linked to every other
#: one, so a whole lobe collapses into a single 6000+ A^2 "patch" that is the
#: protein surface, not an epitope.  A fixed radius is what epitope-prediction
#: methods use, and 10 A reproduces the 15-25 residue / 700-900 A^2 footprint
#: an antibody actually buries.  12 A was chosen by sweeping 10/12/14 A against
#: the paratope target: 10 A under-covers (6-8 res, ~550 A^2) and 14 A spills
#: past a single epitope (16 res, ~1200 A^2).
PATCH_RADIUS = 12.0

#: Two patches describing the same piece of surface are collapsed when they
#: share at least this fraction of their residues.
PATCH_REDUNDANCY = 0.5

#: Antibody paratopes bury ~600-900 A^2.  A patch below the floor cannot host
#: a full paratope; one far above it is really several epitopes' worth.
PARATOPE_SASA_MIN = 500.0
PARATOPE_SASA_IDEAL = 700.0

#: Mean pLDDT below which the local backbone is too uncertain to dock or
#: design against.  A patch can be perfectly free of claimed epitopes and
#: still be useless if the model does not know where its atoms are.
PLDDT_DESIGNABLE_MIN = 70.0

KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "E": -3.5,
    "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}
CHARGED = set("DEKR")


def heavy_atom_coords(pdb_path: Path) -> dict[int, np.ndarray]:
    """Heavy-atom coordinates per residue, keyed by mature residue number."""
    structure = PDBParser(QUIET=True).get_structure("gpc3", str(pdb_path))
    coords: dict[int, np.ndarray] = {}
    for res in structure[0]["A"]:
        if res.id[0] != " ":
            continue
        pts = [a.coord for a in res.get_atoms() if a.element != "H"]
        if pts:
            coords[res.id[1]] = np.array(pts)
    return coords


def min_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min())


def enumerate_patches(
    seed_mature: list[int],
    coords: dict[int, np.ndarray],
    all_exposed: set[int],
) -> list[dict]:
    """Every paratope-sized patch centred on a residue of the seeding region.

    Membership is drawn from ALL exposed residues, not just the region, so a
    patch nucleated in a clean region still reveals the claimed residues it
    leans into -- which is how a conformational epitope actually behaves.
    """
    centroid = {m: coords[m].mean(axis=0) for m in coords}
    exposed_ids = [m for m in sorted(all_exposed) if m in centroid]
    exposed_pts = np.array([centroid[m] for m in exposed_ids])

    patches: list[dict] = []
    for centre in seed_mature:
        if centre not in centroid:
            continue
        within = np.linalg.norm(exposed_pts - centroid[centre], axis=1) <= PATCH_RADIUS
        members = [exposed_ids[i] for i in np.flatnonzero(within)]
        if len(members) < 4:
            continue
        patches.append({"centre": centre, "members": sorted(members)})
    return patches


def deduplicate(patches: list[dict], key) -> list[dict]:
    """Greedily keep the best patch of each overlapping family."""
    kept: list[dict] = []
    for patch in sorted(patches, key=key, reverse=True):
        members = set(patch["members"])
        redundant = False
        for k in kept:
            shared = len(members & set(k["members"]))
            if shared / min(len(members), len(k["members"])) >= PATCH_REDUNDANCY:
                redundant = True
                break
        if not redundant:
            kept.append(patch)
    return kept


def describe(mature_ids: list[int], res_by_mature: dict) -> dict:
    residues = [res_by_mature[m] for m in mature_ids if m in res_by_mature]
    if not residues:
        return {}
    sasa = sum(r.sasa for r in residues)
    return {
        "n_residues": len(residues),
        "sasa": round(sasa, 1),
        "mean_plddt": round(sum(r.plddt for r in residues) / len(residues), 1),
        "hydropathy": round(
            sum(KYTE_DOOLITTLE.get(r.aa, 0) for r in residues) / len(residues), 2
        ),
        "n_charged": sum(1 for r in residues if r.aa in CHARGED),
        "composition": "".join(r.aa for r in residues),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="GPC3_regionB",
                    help="region stem, e.g. GPC3_regionB")
    ap.add_argument("--all", action="store_true", help="run every region")
    ap.add_argument("--top", type=int, default=5, help="patches to report per region")
    ap.add_argument("--radius", type=float, help=f"patch radius in A (default {PATCH_RADIUS})")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if args.radius:
        globals()["PATCH_RADIUS"] = args.radius

    db = json.loads(PATENT_DB.read_text())
    claimed: dict[str, set[int]] = {}
    for ep in db["epitopes"]:
        span = ep.get("uniprot_residues")
        # Skip whole-lobe entries: they are not residue-resolved and would
        # mark every patch in the lobe as encumbered.
        if span and span["end"] - span["start"] > 120:
            continue
        claimed[ep["id"]] = epitope_uniprot_positions(ep)

    full_pdb = STRUCTURE_DIR / "GPC3_ectodomain_full.pdb"
    full = load_residues(full_pdb, with_sasa=True)
    res_by_mature = {r.mature: r for r in full}
    coords = heavy_atom_coords(full_pdb)
    all_exposed = {r.mature for r in full if r.exposed}

    regions = (
        sorted(p.stem for p in STRUCTURE_DIR.glob("GPC3_region*.pdb"))
        if args.all else [args.region]
    )

    output = []
    for stem in regions:
        region_res = load_residues(STRUCTURE_DIR / f"{stem}.pdb")
        region_mature = {r.mature for r in region_res}
        seeds = sorted(region_mature & all_exposed)

        print("=" * 78)
        print(f"{stem}: {len(seeds)} exposed residues, "
              f"patch radius {PATCH_RADIUS} A")
        print("=" * 78)

        candidates = enumerate_patches(seeds, coords, all_exposed)
        scored = []
        for patch in candidates:
            members = patch["members"]
            stats = describe(members, res_by_mature)
            uniprot = {res_by_mature[m].uniprot for m in members if m in res_by_mature}
            collisions = {
                ep_id: sorted(uniprot & positions)
                for ep_id, positions in claimed.items() if uniprot & positions
            }
            in_region = [m for m in members if m in region_mature]
            outside = [m for m in members if m not in region_mature]

            if collisions:
                verdict = "COLLIDES"
            elif stats["sasa"] < PARATOPE_SASA_MIN:
                verdict = "TOO SMALL"
            elif stats["mean_plddt"] < PLDDT_DESIGNABLE_MIN:
                verdict = "LOW CONFIDENCE"
            else:
                verdict = "USABLE"

            scored.append({**patch, "stats": stats, "collisions": collisions,
                           "verdict": verdict, "in_region": in_region,
                           "outside_region": outside,
                           "uniprot": sorted(uniprot)})

        # Rank clean, well-formed, confidently-modelled patches first.
        def quality(p):
            return (
                p["verdict"] == "USABLE",
                len(p["in_region"]) / max(len(p["members"]), 1),
                p["stats"]["mean_plddt"],
            )

        kept = deduplicate(scored, key=quality)[: args.top]
        region_out = {"region": stem, "n_exposed_seeds": len(seeds),
                      "n_candidate_patches": len(scored), "patches": []}

        for i, patch in enumerate(kept, 1):
            st = patch["stats"]
            print(f"\n  Patch {i}  [{patch['verdict']}]  centre mature {patch['centre']} "
                  f"(UniProt {res_by_mature[patch['centre']].uniprot})")
            print(f"    {st['n_residues']} residues, {st['sasa']} A^2, "
                  f"pLDDT {st['mean_plddt']}, hydropathy {st['hydropathy']:+.2f}, "
                  f"{st['n_charged']} charged")
            print(f"    from this region : {patch['in_region']}")
            if patch["outside_region"]:
                print(f"    from elsewhere   : {patch['outside_region']}")
            if patch["collisions"]:
                for ep_id, hits in patch["collisions"].items():
                    print(f"    !! collides with {ep_id} at UniProt {hits}")
            else:
                print("    clean: no residue-resolved claimed epitope in the footprint")
            region_out["patches"].append(patch)

        usable = [p for p in kept if p["verdict"] == "USABLE"]
        print(f"\n  => {len(usable)} usable of {len(kept)} reported "
              f"({len(scored)} candidates before dedup)\n")
        output.append(region_out)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(output, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
