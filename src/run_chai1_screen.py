#!/usr/bin/env python3
"""
run_chai1_screen.py — Chai-1r batch screening with GPC3 hotspot epitope restraints

Screens antibody VH/VL candidates against GPC3 antigen using Chai-1r with
epitope constraints. Outputs ipTM-min across 5 seeds plus per-seed PAE/pLDDT.

Usage:
    python src/run_chai1_screen.py \
        --fasta_dir results/chai1_inputs \
        --antigen_fasta data/gpc3_mature.fasta \
        --out_dir results/chai1_screen \
        --n_seeds 5 \
        --top_k 10

Requirements:
    pip install chai-lab>=0.5.0  # RTX 3060 Ti (8 GB VRAM) or better
    CUDA must be available (chai-lab uses torch CUDA backend)

GPC3 hotspot residues (mature Uniprot numbering, 1-based):
    267, 270, 273, 274, 277, 372, 373, 375, 376, 379, 380
    These are passed as contact_probs=1.0 constraints to Chai-1r.

FASTA input format (one file per candidate, named <candidate_id>.fasta):
    >VH|<candidate_id>
    EVQLVESGG...
    >VL|<candidate_id>
    DIVMTQSPS...

Output:
    results/chai1_screen/
        scores.csv          — all candidates ranked by ipTM_min
        top_k/              — PDB + confidence JSON for top-K candidates
        logs/               — per-candidate stdout/stderr
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── GPC3 epitope definition ────────────────────────────────────────────────────
# Hotspot residues in mature Uniprot numbering (1-based, chain A of antigen).
# Source: validated from structural data in this project.
HOTSPOT_MATURE = [267, 270, 273, 274, 277, 372, 373, 375, 376, 379, 380]

# Chai-1r contact constraint: residue i (antigen) must contact any antibody residue
# with probability >= CONTACT_PROB.  1.0 = hard constraint.
CONTACT_PROB = 1.0

# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class CandidateScore:
    candidate_id: str
    iptm_min: float          # min ipTM across seeds — most conservative metric
    iptm_mean: float
    ptm_mean: float
    plddt_mean: float
    n_seeds_ok: int          # seeds that completed without error
    best_seed: int
    fasta_path: str


# ── FASTA utilities ────────────────────────────────────────────────────────────
def read_fasta(path: Path) -> dict[str, str]:
    """Return {header: sequence} from a FASTA file."""
    seqs: dict[str, str] = {}
    current_header = None
    buf: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current_header is not None:
                    seqs[current_header] = "".join(buf)
                current_header = line[1:]
                buf = []
            else:
                buf.append(line)
    if current_header is not None:
        seqs[current_header] = "".join(buf)
    return seqs


def write_chai_fasta(vh_seq: str, vl_seq: str, antigen_seq: str,
                     out_path: Path, candidate_id: str) -> None:
    """Write combined FASTA for Chai-1r: antigen + VH + VL."""
    with open(out_path, "w") as f:
        f.write(f">protein|name=GPC3_antigen\n{antigen_seq}\n")
        f.write(f">protein|name=VH_{candidate_id}\n{vh_seq}\n")
        f.write(f">protein|name=VL_{candidate_id}\n{vl_seq}\n")


# ── Chai-1r runner ─────────────────────────────────────────────────────────────
def run_chai1_candidate(
    chai_fasta: Path,
    out_dir: Path,
    n_seeds: int,
    hotspot_residues: list[int],
    contact_prob: float,
) -> Optional[CandidateScore]:
    """
    Run Chai-1r for one candidate and return aggregated scores.

    Returns None if chai-lab is not installed or GPU is unavailable.
    """
    try:
        import torch
        from chai_lab.chai1 import run_inference
    except ImportError:
        log.error("chai-lab not installed. Run: pip install chai-lab>=0.5.0")
        return None

    if not torch.cuda.is_available():
        log.error("CUDA not available. Chai-1r requires a CUDA GPU (RTX 3060 Ti+).")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build contact constraints: hotspot residue i (0-based) on chain 0 (antigen)
    # must contact any residue on chains 1 (VH) or 2 (VL).
    # Chai-1r accepts contact constraints as list of dicts.
    contacts = []
    for res in hotspot_residues:
        # res is 1-based mature numbering → 0-based index for Chai API
        contacts.append({
            "chain_1": 0,          # antigen chain index
            "residue_1": res - 1,  # 0-based
            "chain_2": 1,          # VH chain index (any residue)
            "residue_2": None,     # None = any residue in chain
            "contact_prob": contact_prob,
        })
        contacts.append({
            "chain_1": 0,
            "residue_1": res - 1,
            "chain_2": 2,          # VL chain index
            "residue_2": None,
            "contact_prob": contact_prob,
        })

    iptms, ptms, plddts = [], [], []
    seed_pdb_paths = []

    for seed in range(n_seeds):
        seed_out = out_dir / f"seed_{seed}"
        seed_out.mkdir(exist_ok=True)

        try:
            candidates = run_inference(
                fasta_file=chai_fasta,
                output_dir=seed_out,
                num_trunk_recycles=3,
                num_diffn_timesteps=200,
                seed=seed,
                device=torch.device("cuda"),
                use_esm_embeddings=True,
                contact_constraints=contacts if contacts else None,
            )

            # Chai-1r returns a list of CandidateResult objects
            best = candidates[0]  # first is highest-scored
            iptm = float(best.aggregate_score)  # ipTM for this seed
            ptm = float(best.ptm) if hasattr(best, "ptm") else float("nan")
            plddt = float(best.plddt.mean()) if hasattr(best, "plddt") else float("nan")

            iptms.append(iptm)
            ptms.append(ptm)
            plddts.append(plddt)
            seed_pdb_paths.append(seed_out / "pred.model_idx_0.pdb")

        except Exception as exc:
            log.warning(f"  seed {seed} failed: {exc}")
            continue

    if not iptms:
        log.error(f"All seeds failed for {chai_fasta.stem}")
        return None

    best_seed_idx = int(np.argmax(iptms))
    return CandidateScore(
        candidate_id=chai_fasta.stem,
        iptm_min=float(np.min(iptms)),
        iptm_mean=float(np.mean(iptms)),
        ptm_mean=float(np.nanmean(ptms)),
        plddt_mean=float(np.nanmean(plddts)),
        n_seeds_ok=len(iptms),
        best_seed=best_seed_idx,
        fasta_path=str(chai_fasta),
    ), seed_pdb_paths[best_seed_idx]


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Chai-1r batch antibody screening")
    ap.add_argument("--fasta_dir", required=True,
                    help="Directory containing per-candidate FASTA files (VH+VL per file)")
    ap.add_argument("--antigen_fasta", required=True,
                    help="FASTA file with GPC3 antigen sequence (single record)")
    ap.add_argument("--out_dir", default="results/chai1_screen",
                    help="Output directory")
    ap.add_argument("--n_seeds", type=int, default=5,
                    help="Number of diffusion seeds per candidate (default: 5)")
    ap.add_argument("--top_k", type=int, default=10,
                    help="Save PDB structures for top-K candidates (default: 10)")
    ap.add_argument("--no_constraints", action="store_true",
                    help="Disable GPC3 hotspot epitope constraints (ablation)")
    ap.add_argument("--iptm_min_cutoff", type=float, default=0.6,
                    help="Minimum ipTM_min to pass screening (default: 0.6)")
    args = ap.parse_args()

    fasta_dir = Path(args.fasta_dir)
    antigen_fasta = Path(args.antigen_fasta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "top_k").mkdir(exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "chai_inputs").mkdir(exist_ok=True)

    # Load antigen sequence
    antigen_seqs = read_fasta(antigen_fasta)
    antigen_seq = next(iter(antigen_seqs.values()))
    log.info(f"Antigen: {len(antigen_seq)} residues")

    # Gather candidate FASTA files
    candidate_fastas = sorted(fasta_dir.glob("*.fasta")) + sorted(fasta_dir.glob("*.fa"))
    if not candidate_fastas:
        log.error(f"No FASTA files found in {fasta_dir}")
        sys.exit(1)
    log.info(f"Candidates to screen: {len(candidate_fastas)}")

    hotspot = [] if args.no_constraints else HOTSPOT_MATURE
    if hotspot:
        log.info(f"GPC3 hotspot constraints: {hotspot} (contact_prob={CONTACT_PROB})")
    else:
        log.info("Hotspot constraints DISABLED (ablation mode)")

    results: list[CandidateScore] = []
    start_total = time.time()

    for i, cand_fasta in enumerate(candidate_fastas):
        candidate_id = cand_fasta.stem
        log.info(f"[{i+1}/{len(candidate_fastas)}] Screening {candidate_id}")

        # Parse VH/VL from candidate FASTA
        seqs = read_fasta(cand_fasta)
        vh_seq = vl_seq = None
        for header, seq in seqs.items():
            h = header.upper()
            if "VH" in h or header.startswith("VH"):
                vh_seq = seq
            elif "VL" in h or header.startswith("VL"):
                vl_seq = seq

        if vh_seq is None or vl_seq is None:
            log.warning(f"  Could not parse VH/VL from {cand_fasta.name}, skipping")
            continue

        log.info(f"  VH={len(vh_seq)} aa, VL={len(vl_seq)} aa")

        # Write combined FASTA for Chai-1r
        chai_fasta = out_dir / "chai_inputs" / f"{candidate_id}.fasta"
        write_chai_fasta(vh_seq, vl_seq, antigen_seq, chai_fasta, candidate_id)

        # Run Chai-1r
        cand_out = out_dir / candidate_id
        t0 = time.time()
        result = run_chai1_candidate(
            chai_fasta=chai_fasta,
            out_dir=cand_out,
            n_seeds=args.n_seeds,
            hotspot_residues=hotspot,
            contact_prob=CONTACT_PROB,
        )
        elapsed = time.time() - t0

        if result is None:
            log.warning(f"  {candidate_id}: no valid result")
            continue

        score, best_pdb = result
        log.info(
            f"  ipTM_min={score.iptm_min:.3f}  ipTM_mean={score.iptm_mean:.3f}"
            f"  pLDDT={score.plddt_mean:.1f}  ({elapsed:.0f}s)"
        )
        results.append(score)

        # Copy best-seed PDB to output
        if best_pdb and best_pdb.exists():
            shutil.copy(best_pdb, cand_out / f"{candidate_id}_best.pdb")

    # ── Rank and save ──────────────────────────────────────────────────────────
    results.sort(key=lambda r: r.iptm_min, reverse=True)

    scores_csv = out_dir / "scores.csv"
    fieldnames = list(asdict(results[0]).keys()) if results else []
    with open(scores_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    log.info(f"Scores saved → {scores_csv}")

    # Copy top-K structures
    for rank, r in enumerate(results[:args.top_k]):
        src = Path(r.fasta_path).parent.parent / r.candidate_id / f"{r.candidate_id}_best.pdb"
        if src.exists():
            dst = out_dir / "top_k" / f"rank{rank+1:02d}_{r.candidate_id}.pdb"
            shutil.copy(src, dst)

    # Summary
    passed = [r for r in results if r.iptm_min >= args.iptm_min_cutoff]
    log.info("=" * 60)
    log.info(f"Screened: {len(results)} candidates  |  Passed (ipTM_min≥{args.iptm_min_cutoff}): {len(passed)}")
    log.info(f"Total time: {(time.time()-start_total)/60:.1f} min")
    log.info("Top 5 by ipTM_min:")
    for rank, r in enumerate(results[:5]):
        flag = "✓" if r.iptm_min >= args.iptm_min_cutoff else " "
        log.info(f"  {flag} {rank+1}. {r.candidate_id:30s}  ipTM_min={r.iptm_min:.3f}  pLDDT={r.plddt_mean:.1f}")

    # Save summary JSON
    summary = {
        "n_candidates": len(results),
        "n_passed": len(passed),
        "iptm_min_cutoff": args.iptm_min_cutoff,
        "hotspot_constraints": hotspot,
        "n_seeds": args.n_seeds,
        "top_k": [asdict(r) for r in results[:args.top_k]],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
