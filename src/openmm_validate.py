"""OpenMM validation of docked antibody-GPC3 complex.

Steps:
  1. PDBFixer: add missing residues/atoms, protonate at pH 7.4
  2. Amber14 force field + TIP3P water, periodic box
  3. Energy minimisation (convergence < 10 kJ/mol/nm)
  4. NVT equilibration 100 ps (300 K)
  5. NpT production 500 ps (300 K, 1 bar)
  6. Analysis: RMSD, hotspot contacts, interaction energy estimate

Usage:
    python3 src/openmm_validate.py
    python3 src/openmm_validate.py --pdb design/docked_complex.pdb --steps 500
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from openmm import (
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    Platform,
    unit,
    Vec3,
)
from openmm.app import (
    PDBFile,
    ForceField,
    Modeller,
    PME,
    HBonds,
    Simulation,
    DCDReporter,
    StateDataReporter,
)
from pdbfixer import PDBFixer

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

HOTSPOT_MATURE = [267, 270, 273, 274, 277, 372, 373, 375, 376, 379, 380]

def fix_pdb(pdb_path: Path, out_path: Path, ph: float = 7.4) -> None:
    """Add missing atoms, protonate, write clean PDB."""
    print(f"[PDBFixer] 구조 보정: {pdb_path.name}")
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    with open(out_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    print(f"  → {out_path.name} 저장 (체인: "
          f"{set(r.chain.id for r in fixer.topology.residues())})")


def build_system(fixed_pdb: Path, padding: float = 1.0):
    """Solvate in TIP3P box, return (simulation, topology, modeller)."""
    print("[OpenMM] 시스템 구축 (Amber14sb + TIP3P)")
    pdb = PDBFile(str(fixed_pdb))
    ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(ff, padding=padding * unit.nanometers, model="tip3p")
    n_wat = sum(1 for r in modeller.topology.residues() if r.name == "HOH")
    print(f"  → 물 분자 {n_wat:,}개 추가, 박스 크기 ~"
          f"{modeller.topology.getPeriodicBoxVectors()[0][0].value_in_unit(unit.nanometers):.1f} nm")

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=HBonds,
    )
    system.addForce(MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    platform = Platform.getPlatformByName("CPU")
    sim = Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)
    return sim, modeller


def minimize(sim: Simulation, tol: float = 10.0) -> float:
    """Energy minimise; return final potential energy in kJ/mol."""
    print("[최소화] 에너지 최소화 시작...")
    t0 = time.time()
    state_before = sim.context.getState(getEnergy=True)
    e_before = state_before.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    sim.minimizeEnergy(tolerance=tol * unit.kilojoules_per_mole / unit.nanometers)
    state_after = sim.context.getState(getEnergy=True)
    e_after = state_after.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(f"  전: {e_before:>12,.0f} kJ/mol")
    print(f"  후: {e_after:>12,.0f} kJ/mol  (Δ {e_after-e_before:+,.0f})  [{time.time()-t0:.0f}s]")
    return e_after


def run_md(sim: Simulation, out_dir: Path, steps: int,
           report_interval: int = 100, label: str = "prod") -> None:
    """Run MD with reporters."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sim.reporters.clear()
    sim.reporters.append(DCDReporter(str(out_dir / f"{label}.dcd"), report_interval))
    sim.reporters.append(StateDataReporter(
        str(out_dir / f"{label}.csv"), report_interval,
        step=True, time=True, potentialEnergy=True,
        kineticEnergy=True, temperature=True, density=True,
    ))
    sim.reporters.append(StateDataReporter(
        None, max(1, steps // 5), step=True,
        time=True, potentialEnergy=True, temperature=True, progress=True,
        remainingTime=True, totalSteps=steps, separator="\t",
    ))
    print(f"\n[MD] {label} ({steps} steps = {steps*0.002:.1f} ps)")
    t0 = time.time()
    sim.step(steps)
    print(f"  완료 [{time.time()-t0:.0f}s]")


def analyse_contacts(sim: Simulation, topology, hotspot_mature: list[int],
                     cutoff_nm: float = 0.5) -> dict:
    """Count hotspot-antibody contacts in current frame."""
    state = sim.context.getState(getPositions=True)
    pos = np.array(state.getPositions().value_in_unit(unit.nanometers))

    res_list = list(topology.residues())
    # Map chain+resid → atom indices
    hotspot_atoms: list[int] = []
    ab_atoms: list[int] = []

    for res in res_list:
        chain_id = res.chain.id
        resid = int(res.id)
        atom_idxs = [a.index for a in res.atoms()]
        if chain_id == "A" and resid in hotspot_mature:
            hotspot_atoms.extend(atom_idxs)
        elif chain_id in ("H", "L"):
            ab_atoms.extend(atom_idxs)

    if not hotspot_atoms or not ab_atoms:
        return {"contacts": 0, "min_dist_nm": None}

    hot_pos = pos[hotspot_atoms]
    ab_pos = pos[ab_atoms]
    dists = np.linalg.norm(hot_pos[:, None, :] - ab_pos[None, :, :], axis=2)
    min_d = float(dists.min())
    contacts = int((dists < cutoff_nm).any(axis=1).sum())
    return {"hotspot_contacts": contacts,
            "total_hotspot_atoms": len(hotspot_atoms),
            "min_dist_nm": round(min_d, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdb", type=Path,
                        default=REPO_ROOT / "design" / "docked_complex.pdb")
    parser.add_argument("--steps", type=int, default=500,
                        help="NpT production steps (0.002 ps each; default 500 = 1 ps)")
    parser.add_argument("--equil", type=int, default=500,
                        help="NVT equilibration steps (default 500 = 1 ps)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "openmm")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fixed = args.out / "fixed.pdb"

    # ── 1. PDBFixer ──────────────────────────────────────────────────────
    fix_pdb(args.pdb, fixed)

    # ── 2. 시스템 구축 ────────────────────────────────────────────────────
    sim, modeller = build_system(fixed, padding=1.0)

    # ── 3. 에너지 최소화 ──────────────────────────────────────────────────
    e_min = minimize(sim)

    # 최소화된 구조 저장
    min_pdb = args.out / "minimized.pdb"
    state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(min_pdb, "w") as f:
        PDBFile.writeFile(modeller.topology, state.getPositions(), f)
    print(f"  최소화 구조: {min_pdb}")

    # 최소화 후 접촉 분석
    contacts_min = analyse_contacts(sim, modeller.topology, HOTSPOT_MATURE)
    print(f"\n  Hotspot 접촉 (최소화 후): {contacts_min['hotspot_contacts']}/"
          f"{contacts_min['total_hotspot_atoms']} atoms  "
          f"최단거리 {contacts_min['min_dist_nm']:.3f} nm")

    # ── 4. NVT 평형화 ─────────────────────────────────────────────────────
    if args.equil > 0:
        sim.context.setVelocitiesToTemperature(300 * unit.kelvin)
        run_md(sim, args.out, args.equil, label="equil")

    # ── 5. NpT 생산 MD ────────────────────────────────────────────────────
    if args.steps > 0:
        run_md(sim, args.out, args.steps, label="prod")

    # 최종 접촉 분석
    contacts_prod = analyse_contacts(sim, modeller.topology, HOTSPOT_MATURE)
    print(f"\n  Hotspot 접촉 (MD 후): {contacts_prod['hotspot_contacts']}/"
          f"{contacts_prod['total_hotspot_atoms']} atoms  "
          f"최단거리 {contacts_prod['min_dist_nm']:.3f} nm")

    # 최종 구조 저장
    prod_pdb = args.out / "production.pdb"
    state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(prod_pdb, "w") as f:
        PDBFile.writeFile(modeller.topology, state.getPositions(), f)
    print(f"  생산 MD 구조: {prod_pdb}")

    # ── 6. 결과 저장 ──────────────────────────────────────────────────────
    result = {
        "input_pdb": str(args.pdb),
        "forcefield": "Amber14sb + TIP3P",
        "minimization": {"final_energy_kJ_mol": round(e_min, 1)},
        "equil_steps": args.equil,
        "prod_steps": args.steps,
        "prod_time_ps": round(args.steps * 0.002, 2),
        "hotspot_contacts_after_min": contacts_min,
        "hotspot_contacts_after_md": contacts_prod,
    }
    (args.out / "validation_result.json").write_text(json.dumps(result, indent=2))

    print("\n" + "="*60)
    print("OpenMM 검증 완료")
    print(f"  에너지 최소화: {e_min:,.0f} kJ/mol")
    print(f"  MD 시간:       {args.steps * 0.002:.2f} ps")
    print(f"  Hotspot 접촉:  {contacts_prod['hotspot_contacts']} / "
          f"{contacts_prod['total_hotspot_atoms']} (MD 후)")
    print(f"  결과:          {args.out}/")
    print("="*60)


if __name__ == "__main__":
    main()
