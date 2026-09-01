"""Prepare structure-based scFv design inputs against a chosen GPC3 patch.

The design run itself needs a GPU and model weights that this sandbox has
neither of, so this does everything that comes before it and everything that
has to be right for the run to mean anything:

  * writes an epitope-focused target PDB (the patch plus the structural
    context around it) so the design job is not paying for the whole
    ectodomain, most of which is a disordered C-terminal linker;
  * writes the hotspot residue specification in the chain+number form the
    RFdiffusion / RFantibody configs expect, taken from the patch the patent
    scan cleared -- so the design is aimed at the site the analysis chose,
    not at whatever the docking software finds most convenient;
  * assembles the scFv scaffold from IMGT human germline sequences, with the
    CDR loops marked as the positions the design run fills in;
  * emits a run script for the GPU machine.

Germline sequences come from the IMGT reference data shipped inside the
`sadie-antibody` package, not from recall.

Usage:
    python3 src/scfv_design_prep.py --patch-centre 273
    python3 src/scfv_design_prep.py --patch-centre 273 --outdir design/patchB2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO, PDBParser, Select

from gpc3_lib import REPO_ROOT, STRUCTURE_DIR, load_residues
from surface_patches import PATCH_RADIUS, heavy_atom_coords

# IMGT-gapped germline alignments are a fixed 128 columns, so CDR boundaries
# are constant column ranges rather than something to search for.
IMGT_CDR_SPANS = {"CDR1": (27, 38), "CDR2": (56, 65), "CDR3": (105, 117)}

#: Most-used human germline pair in synthetic scFv libraries.  Both are
#: unmutated germline, which is the cleanest possible starting framework.
DEFAULT_VH_V, DEFAULT_VH_J = "IGHV3-23*01", "IGHJ4*01"
DEFAULT_VL_V, DEFAULT_VL_J = "IGKV1-39*01", "IGKJ1*01"

#: Trastuzumab (hu4D5) variable domains -- the most-used scFv/CAR framework
#: there is, picked for its expression and thermal stability rather than for
#: its HER2 CDRs, which a design run replaces.
#:
#: Taken from public repositories that agree character for character
#: (AbSciBio/unlocking-de-novo-antibody-design, RosettaCommons/FvHallucinator,
#: amhummer/Graphinity, prescient-design/ibex, snijderlab/stitch), and the
#: CDR split is re-derived here by IMGT numbering rather than trusted from
#: any one of them.
TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKG"
    "RFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSR"
    "SGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)

#: (G4S)x3 -- long enough to let VH and VL pair without strain, short enough
#: to disfavour the diabody form that shorter linkers force.
LINKER = "GGGGS" * 3

#: Residues within this distance of the patch are kept in the trimmed target,
#: so the designed binder sees the real local environment, not a floating
#: fragment.
CONTEXT_RADIUS = 16.0


def germline(kind: str, chain: str, name: str) -> str:
    from sadie.numbering.germlines import all_germlines

    try:
        return all_germlines[kind][chain]["human"][name]
    except KeyError as exc:
        raise SystemExit(f"germline {name} not found in the IMGT reference") from exc


def cdr_positions(gapped: str) -> dict[str, list[int]]:
    """Ungapped indices of each CDR within a gapped germline alignment."""
    out: dict[str, list[int]] = {}
    for cdr, (lo, hi) in IMGT_CDR_SPANS.items():
        idx, ungapped_i = [], 0
        for col, aa in enumerate(gapped, start=1):
            if aa == "-":
                continue
            ungapped_i += 1
            if lo <= col <= hi:
                idx.append(ungapped_i)
        if idx:
            out[cdr] = idx
    return out


def build_chain(v_name: str, j_name: str, chain: str) -> dict:
    v_gapped = germline("V", chain, v_name)
    j_gapped = germline("J", chain, j_name)
    v_seq = v_gapped.replace("-", "")
    j_seq = j_gapped.replace("-", "")
    return {
        "v_gene": v_name,
        "j_gene": j_name,
        "v_sequence": v_seq,
        "j_sequence": j_seq,
        "scaffold": v_seq + j_seq,
        "cdr_positions_in_v": cdr_positions(v_gapped),
        "cdr1": "".join(v_seq[i - 1] for i in cdr_positions(v_gapped).get("CDR1", [])),
        "cdr2": "".join(v_seq[i - 1] for i in cdr_positions(v_gapped).get("CDR2", [])),
    }


def split_framework(seq: str, label: str) -> dict:
    """Split a variable domain into framework and CDR loops by IMGT numbering.

    The CDR positions are what a design run may change; everything else is the
    framework the user asked to keep, so an off-by-one here silently freezes
    the wrong residues.  This walks ANARCI's raw IMGT numbering rather than
    using abnumber's cdrN_seq helpers, which in the build installed here slice
    the kappa chain one residue to the left (VL CDR2 comes back as "YSA"
    instead of "SAS").
    """
    from anarci import run_anarci

    numbering = run_anarci([(label, seq)], scheme="imgt")[1][0]
    if not numbering:
        raise SystemExit(f"ANARCI could not number {label}")
    domain, start, _end = numbering[0]

    # Walk the numbering in order, tracking where each numbered residue sits
    # in the query.  Gaps carry no query residue and must not advance it.
    cursor = start
    positions: dict[int, int] = {}
    for (imgt_pos, _insertion), aa in domain:
        if aa == "-":
            continue
        cursor += 1
        positions.setdefault(imgt_pos, cursor)  # 1-based in the query

    cdrs, spans = {}, {}
    for name, (lo, hi) in IMGT_CDR_SPANS.items():
        residues = [
            (positions[p], aa)
            for (p, _i), aa in domain
            if aa != "-" and lo <= p <= hi and p in positions
        ]
        if not residues:
            continue
        cdrs[name] = "".join(aa for _, aa in residues)
        spans[name] = [residues[0][0], residues[-1][0]]

    return {
        "sequence": seq,
        "length": len(seq),
        "cdr1": cdrs.get("CDR1", ""),
        "cdr2": cdrs.get("CDR2", ""),
        "cdr3": cdrs.get("CDR3", ""),
        "cdr_spans_1based": spans,
        "design_positions": sorted(
            i for span in spans.values() for i in range(span[0], span[1] + 1)
        ),
    }


class PatchContext(Select):
    def __init__(self, keep: set[int]):
        self.keep = keep

    def accept_residue(self, residue):
        return residue.id[0] == " " and residue.id[1] in self.keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch-centre", type=int, default=273,
                    help="mature residue number at the centre of the target patch")
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "design")
    ap.add_argument("--vh-v", default=DEFAULT_VH_V)
    ap.add_argument("--vh-j", default=DEFAULT_VH_J)
    ap.add_argument("--vl-v", default=DEFAULT_VL_V)
    ap.add_argument("--vl-j", default=DEFAULT_VL_J)
    ap.add_argument("--orientation", choices=["VH-VL", "VL-VH"], default="VH-VL")
    ap.add_argument("--framework", choices=["trastuzumab", "germline"],
                    default="trastuzumab",
                    help="trastuzumab (hu4D5) scaffold, or bare human germline")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    full_pdb = STRUCTURE_DIR / "GPC3_ectodomain_full.pdb"
    full = load_residues(full_pdb, with_sasa=True)
    by_mature = {r.mature: r for r in full}
    coords = heavy_atom_coords(full_pdb)
    exposed = {r.mature for r in full if r.exposed}

    centre = args.patch_centre
    if centre not in coords:
        raise SystemExit(f"residue {centre} not present in the model")

    centroid = {m: coords[m].mean(axis=0) for m in coords}
    hotspots = sorted(
        m for m in exposed
        if np.linalg.norm(centroid[m] - centroid[centre]) <= PATCH_RADIUS
    )
    context = sorted(
        m for m in coords
        if np.linalg.norm(centroid[m] - centroid[centre]) <= CONTEXT_RADIUS
    )

    # --- epitope-focused target ------------------------------------------
    structure = PDBParser(QUIET=True).get_structure("gpc3", str(full_pdb))
    io = PDBIO()
    io.set_structure(structure)
    target_pdb = outdir / f"target_patch{centre}.pdb"
    io.save(str(target_pdb), PatchContext(set(context)))

    # --- scFv scaffold ----------------------------------------------------
    if args.framework == "trastuzumab":
        vh = {"source": "trastuzumab (hu4D5) VH", **split_framework(TRASTUZUMAB_VH, "VH")}
        vl = {"source": "trastuzumab (hu4D5) VL", **split_framework(TRASTUZUMAB_VL, "VL")}
        vh["scaffold"], vl["scaffold"] = vh["sequence"], vl["sequence"]
    else:
        vh = build_chain(args.vh_v, args.vh_j, "H")
        vl = build_chain(args.vl_v, args.vl_j, "K")
        vh["source"] = f"{args.vh_v} + {args.vh_j}"
        vl["source"] = f"{args.vl_v} + {args.vl_j}"
    first, second = (vh, vl) if args.orientation == "VH-VL" else (vl, vh)
    scfv = first["scaffold"] + LINKER + second["scaffold"]

    # Where the designable loops sit in the assembled single chain, so the
    # design run can be told what to hold fixed.
    offset_second = len(first["scaffold"]) + len(LINKER)
    design_positions_scfv = (
        [i for i in first.get("design_positions", [])]
        + [i + offset_second for i in second.get("design_positions", [])]
    )

    spec = {
        "target": {
            "antigen": "GPC3 / Glypican-3 (UniProt P51654)",
            "structure": str(target_pdb.relative_to(REPO_ROOT)),
            "numbering": "mature chain; uniprot = mature + 24",
            "patch_centre_mature": centre,
            "patch_centre_uniprot": by_mature[centre].uniprot,
            "hotspot_mature": hotspots,
            "hotspot_uniprot": [by_mature[m].uniprot for m in hotspots],
            "hotspot_res": [f"A{m}" for m in hotspots],
            "context_residues_mature": context,
            "mean_plddt_hotspots": round(
                sum(by_mature[m].plddt for m in hotspots) / len(hotspots), 1),
            "patch_sasa": round(sum(by_mature[m].sasa for m in hotspots), 1),
        },
        "scfv": {
            "framework": args.framework,
            "orientation": args.orientation,
            "linker": LINKER,
            "heavy": vh,
            "light": vl,
            "scaffold_sequence": scfv,
            "scaffold_length": len(scfv),
            "design_positions_in_scfv": design_positions_scfv,
            "note": "design_positions_in_scfv are the six CDR loops, 1-based in "
                    "the assembled single chain. Everything else is the "
                    "framework and is held fixed.",
        },
        "sequence_provenance": (
            "trastuzumab VH/VL from cross-agreeing public repositories, CDR "
            "boundaries re-derived by IMGT numbering (ANARCI/abnumber); "
            "germline option from IMGT data bundled with sadie-antibody"
        ),
    }
    (outdir / "design_spec.json").write_text(json.dumps(spec, indent=2))

    fasta = outdir / "scfv_scaffold.fasta"
    fasta.write_text(
        f">scFv_scaffold_{args.orientation}_{args.vh_v}_{args.vl_v}\n{scfv}\n")

    # --- run script -------------------------------------------------------
    hotspot_arg = ",".join(f"A{m}" for m in hotspots)
    (outdir / "run_design.sh").write_text(f"""#!/usr/bin/env bash
# GPC3 scFv design -- run on a GPU machine with RFantibody installed.
# Generated by src/scfv_design_prep.py; do not hand-edit the hotspots,
# regenerate them so they stay tied to the patent scan.
set -euo pipefail

TARGET="{target_pdb.name}"
HOTSPOTS="[{hotspot_arg}]"
NDESIGN="${{NDESIGN:-200}}"

# 1. backbone design: dock a VHH/scFv framework onto the epitope
poetry run python /opt/RFantibody/scripts/rfdiffusion_inference.py \\
    --config-name antibody \\
    antibody.target_pdb="$TARGET" \\
    antibody.framework_pdb=framework.pdb \\
    inference.ckpt_override_path=/opt/RFantibody/weights/RFdiffusion_Ab.pt \\
    'ppi.hotspot_res='"$HOTSPOTS" \\
    'antibody.design_loops=[L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13]' \\
    inference.num_designs="$NDESIGN" \\
    inference.output_prefix=out/backbones/design

# 2. sequence design on the docked backbones
poetry run python /opt/RFantibody/scripts/proteinmpnn_interface_design.py \\
    -pdbdir out/backbones -outpdbdir out/seqs

# 3. structure prediction filter -- keep designs that fold as intended
poetry run python /opt/RFantibody/scripts/rf2_predict.py \\
    input.pdb_dir=out/seqs output.pdb_dir=out/predicted

# 4. patent re-check on the surviving CDRs (runs anywhere, no GPU)
python3 ../src/patent_seq_similarity.py query --fasta out/predicted/cdrs.fasta
""")
    (outdir / "run_design.sh").chmod(0o755)

    # --- report -----------------------------------------------------------
    print("=" * 78)
    print(f"scFv design inputs for patch centred on mature {centre} "
          f"(UniProt {by_mature[centre].uniprot})")
    print("=" * 78)
    print(f"\ntarget")
    print(f"  hotspots      : {len(hotspots)} residues, {spec['target']['patch_sasa']} A^2, "
          f"mean pLDDT {spec['target']['mean_plddt_hotspots']}")
    print(f"  mature        : {hotspots}")
    print(f"  uniprot       : {spec['target']['hotspot_uniprot']}")
    print(f"  hotspot_res   : [{hotspot_arg}]")
    print(f"  trimmed target: {target_pdb.name} ({len(context)} residues "
          f"within {CONTEXT_RADIUS} A)")
    print(f"\nscFv scaffold ({args.framework}, {args.orientation}, {len(scfv)} aa)")
    for label, ch in (("heavy", vh), ("light", vl)):
        print(f"  {label} : {ch['source']}")
        print(f"          CDR1={ch['cdr1']}  CDR2={ch['cdr2']}  "
              f"CDR3={ch.get('cdr3', '(designed)')}")
    print(f"  linker: {LINKER}")
    print(f"  design positions (CDRs) in the scFv: "
          f"{len(design_positions_scfv)} of {len(scfv)} residues")
    print(f"\n  {scfv}")
    print(f"\nwrote {outdir}/")
    for f in sorted(outdir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
