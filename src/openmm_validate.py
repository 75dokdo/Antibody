"""OpenMM MD validation of docked antibody-GPC3 complex.

Steps:
  1. PDBFixer: add missing residues/atoms, protonate at pH 7.4
  2. Force field + water box + 0.15 M NaCl
       ff14SB mode (기본): Amber14sb + TIP3P-FB
       ff19SB mode:        Amber ff19SB + OPC (4-점, 권장)
  3. Staged Cα-restrained energy minimisation (k = 1000 → 100 → 10 → 0 kJ/mol/nm²)
  4. NVT equilibration (default 50,000 steps = 100 ps, 300 K)
  5. NpT production MD (default 250,000 steps = 500 ps, 300 K, 1 bar)
  6. MDTraj analysis: RMSD, RMSF, PBC-corrected centroid distance,
     hotspot-contact time series, B-factor, radius of gyration

Usage:
    # ff14SB (빠른 검증)
    python3 src/openmm_validate.py --pdb design/docked_complex.pdb

    # ff19SB + OPC (정밀 MD)
    python3 src/openmm_validate.py --pdb design/docked_complex.pdb \\
        --ff ff19sb --equil 50000 --steps 250000

    # 기존 DCD만 재분석
    python3 src/openmm_validate.py --analysis_only --out results/openmm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from openmm import (
    CustomExternalForce,
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

# ── 포스필드 설정 ─────────────────────────────────────────────────────────────
FF_CONFIGS = {
    "ff14sb": {
        "ff_files":    ["amber14-all.xml", "amber14/tip3pfb.xml"],
        "water_model": "tip3p",
        "cutoff_nm":   1.0,
        "label":       "Amber14sb + TIP3P-FB",
    },
    "ff19sb": {
        # ff19SB는 OPC 워터와 함께 설계됨 (Tian et al. 2020)
        # OPC: 4-점 모델, 물 성질 최고 재현 (Izadi et al. 2014)
        "ff_files":    ["amber/ff19SB.xml", "amber/opc.xml"],
        "water_model": "opc",
        "cutoff_nm":   0.9,   # OPC 권장 컷오프
        "label":       "Amber ff19SB + OPC",
    },
}

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

HOTSPOT_MATURE = [267, 270, 273, 274, 277, 372, 373, 375, 376, 379, 380]


def map_hotspots_to_sequential(pdb_path: Path,
                                hotspot_mature: list[int]) -> list[int]:
    """Convert mature-sequence residue numbers to sequential 1-based indices.

    PDBFixer renumbers chain A residues from the original (gapped) mature
    numbering to sequential 1-53. This function reads the ORIGINAL PDB to
    build the mapping before renumbering occurs.
    """
    chain_a_resnums: list[int] = []
    seen: set[int] = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[21] == "A":
                rn = int(line[22:26].strip())
                if rn not in seen:
                    chain_a_resnums.append(rn)
                    seen.add(rn)
    chain_a_resnums.sort()
    hotspot_set = set(hotspot_mature)
    seq_ids = [i for i, rn in enumerate(chain_a_resnums, start=1)
               if rn in hotspot_set]
    # Print mapping for diagnostics
    print("  Hotspot 잔기 매핑 (mature → sequential):")
    for i, rn in enumerate(chain_a_resnums, start=1):
        if rn in hotspot_set:
            print(f"    mature {rn} → seq {i}")
    return seq_ids


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


def build_system(fixed_pdb: Path, ff_key: str = "ff14sb",
                 padding: float = 1.0, salt_molar: float = 0.15,
                 temperature_K: float = 300.0, dt_fs: float = 2.0):
    """Solvate, add 0.15 M NaCl, build OpenMM Simulation.

    Args:
        ff_key:      'ff14sb' or 'ff19sb'
        padding:     Water box padding in nm
        salt_molar:  NaCl concentration in mol/L (default 0.15 M physiological)
        temperature_K: Simulation temperature
        dt_fs:       Integration timestep in fs

    Returns:
        (simulation, modeller)
    """
    cfg = FF_CONFIGS.get(ff_key, FF_CONFIGS["ff14sb"])
    print(f"[OpenMM] 시스템 구축 ({cfg['label']})")
    pdb = PDBFile(str(fixed_pdb))
    ff = ForceField(*cfg["ff_files"])
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(
        ff,
        padding=padding * unit.nanometers,
        model=cfg["water_model"],
        ionicStrength=salt_molar * unit.molar,   # 0.15 M NaCl 생리 농도
        positiveIon="Na+",
        negativeIon="Cl-",
    )
    n_wat = sum(1 for r in modeller.topology.residues() if r.name == "HOH")
    box_nm = modeller.topology.getPeriodicBoxVectors()[0][0].value_in_unit(unit.nanometers)
    print(f"  → 물 분자 {n_wat:,}개 / NaCl 0.15 M / 박스 ~{box_nm:.1f} nm")
    print(f"  → 비결합 컷오프: {cfg['cutoff_nm']} nm")

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=cfg["cutoff_nm"] * unit.nanometers,
        constraints=HBonds,
        rigidWater=True,
        ewaldErrorTolerance=5e-4,
    )
    # NpT: Monte Carlo barostat
    system.addForce(MonteCarloBarostat(
        1 * unit.bar, temperature_K * unit.kelvin
    ))

    integrator = LangevinMiddleIntegrator(
        temperature_K * unit.kelvin,
        1.0 / unit.picosecond,
        dt_fs * unit.femtoseconds,
    )

    # GPU 우선, 없으면 CPU
    try:
        platform = Platform.getPlatformByName("CUDA")
        props = {"CudaPrecision": "mixed"}
        sim = Simulation(modeller.topology, system, integrator, platform, props)
        print("  → 플랫폼: CUDA (GPU)")
    except Exception:
        try:
            platform = Platform.getPlatformByName("OpenCL")
            sim = Simulation(modeller.topology, system, integrator, platform)
            print("  → 플랫폼: OpenCL (GPU/Apple Metal)")
        except Exception:
            platform = Platform.getPlatformByName("CPU")
            sim = Simulation(modeller.topology, system, integrator, platform)
            print("  → 플랫폼: CPU")

    sim.context.setPositions(modeller.positions)
    return sim, modeller


def add_ca_restraints(sim: "Simulation", modeller: "Modeller",
                      k_kj: float = 1000.0) -> "CustomExternalForce":
    """Add harmonic Cα positional restraints to the system.

    Restraint energy: E = k * [(x-x0)^2 + (y-y0)^2 + (z-z0)^2]
    Returns the force object so k can be updated via setGlobalParameterDefaultValue.
    """
    restraint = CustomExternalForce(
        "k_ca*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    restraint.addGlobalParameter(
        "k_ca", k_kj * unit.kilojoules_per_mole / unit.nanometers**2
    )
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    positions = modeller.positions
    n_ca = 0
    for atom in modeller.topology.atoms():
        # Restrain all Cα atoms in protein chains (not water/ions)
        if atom.name == "CA" and atom.residue.name not in ("HOH", "WAT", "CL", "NA"):
            x0, y0, z0 = positions[atom.index].value_in_unit(unit.nanometers)
            restraint.addParticle(atom.index, [x0, y0, z0])
            n_ca += 1

    sim.system.addForce(restraint)
    sim.context.reinitialize(preserveState=True)
    print(f"  → Cα 구속 추가: {n_ca}개 원자, k={k_kj:.0f} kJ/mol/nm²")
    return restraint


def update_restraint_k(sim: "Simulation", restraint: "CustomExternalForce",
                        k_kj: float) -> None:
    """Update the restraint force constant without reinitializing."""
    restraint.setGlobalParameterDefaultValue(
        0, k_kj * unit.kilojoules_per_mole / unit.nanometers**2
    )
    sim.context.setParameter(
        "k_ca", k_kj * unit.kilojoules_per_mole / unit.nanometers**2
    )


def minimize(sim: "Simulation", tol: float = 10.0, label: str = "") -> float:
    """Energy minimise; return final potential energy in kJ/mol."""
    tag = f"[최소화{' '+label if label else ''}]"
    print(f"{tag} 에너지 최소화 시작...")
    t0 = time.time()
    state_before = sim.context.getState(getEnergy=True)
    e_before = state_before.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    sim.minimizeEnergy(tolerance=tol * unit.kilojoules_per_mole / unit.nanometers)
    state_after = sim.context.getState(getEnergy=True)
    e_after = state_after.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(f"  전: {e_before:>12,.0f} kJ/mol")
    print(f"  후: {e_after:>12,.0f} kJ/mol  (Δ {e_after-e_before:+,.0f})  [{time.time()-t0:.0f}s]")
    return e_after


def run_md(sim: "Simulation", out_dir: Path, steps: int,
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


def _get_chain_roles(topology) -> dict:
    """Return {'antigen': chain_id, 'antibody': [chain_ids]} by chain index.

    OpenMM renumbers chains alphabetically after solvation (H→B, L→C).
    We use chain ORDER: index-0=antigen(A), index-1=VH, index-2=VL.
    Water/ion chains (residues named HOH/WAT/CL/NA) are excluded.
    """
    protein_chains = []
    water_names = {"HOH", "WAT", "CL", "NA", "K", "MG", "CA"}
    for chain in topology.chains():
        res_names = {r.name for r in chain.residues()}
        if res_names - water_names:  # has at least one non-water residue
            protein_chains.append(chain.id)
    # index-0 = antigen, index-1,2 = antibody chains
    return {
        "antigen": protein_chains[0] if protein_chains else None,
        "antibody": protein_chains[1:] if len(protein_chains) > 1 else [],
    }


def analyse_contacts(sim: "Simulation", topology, hotspot_mature: list[int],
                     cutoff_nm: float = 0.5) -> dict:
    """Count hotspot-antibody contacts in current frame.

    Robust to OpenMM chain renaming (H→B, L→C after solvation).
    Hotspot residues are found by residue number in the antigen chain.
    """
    state = sim.context.getState(getPositions=True)
    pos = np.array(state.getPositions().value_in_unit(unit.nanometers))

    roles = _get_chain_roles(topology)
    ag_chain = roles["antigen"]
    ab_chains = set(roles["antibody"])

    hotspot_atoms: list[int] = []
    ab_atoms: list[int] = []
    ag_resids: list[int] = []

    for res in topology.residues():
        chain_id = res.chain.id
        resid = int(res.id)
        atom_idxs = [a.index for a in res.atoms()]
        if chain_id == ag_chain:
            ag_resids.append(resid)
            if resid in hotspot_mature:
                hotspot_atoms.extend(atom_idxs)
        elif chain_id in ab_chains:
            ab_atoms.extend(atom_idxs)

    if not hotspot_atoms or not ab_atoms:
        chains_found = {r.chain.id for r in topology.residues()}
        hotspot_set = set(hotspot_mature)
        ag_set = set(ag_resids)
        print(f"  [주의] hotspot 또는 항체 원자 미탐지.")
        print(f"    발견 체인: {chains_found}")
        print(f"    항원 체인({ag_chain}) 잔기 범위: "
              f"{min(ag_resids) if ag_resids else '?'} – {max(ag_resids) if ag_resids else '?'}")
        print(f"    HOTSPOT_MATURE 교집합: {hotspot_set & ag_set}")
        print(f"    항체 체인: {ab_chains}")
        return {"hotspot_contacts": 0, "total_hotspot_atoms": 0, "min_dist_nm": None}

    hot_pos = pos[hotspot_atoms]
    ab_pos = pos[ab_atoms]
    dists = np.linalg.norm(hot_pos[:, None, :] - ab_pos[None, :, :], axis=2)
    min_d = float(dists.min())
    contacts = int((dists < cutoff_nm).any(axis=1).sum())
    return {"hotspot_contacts": contacts,
            "total_hotspot_atoms": len(hotspot_atoms),
            "min_dist_nm": round(min_d, 3)}


def centroid_distance(sim: "Simulation", topology) -> float:
    """Return centroid-to-centroid distance (nm) between antigen and antibody chains."""
    state = sim.context.getState(getPositions=True)
    pos = np.array(state.getPositions().value_in_unit(unit.nanometers))
    roles = _get_chain_roles(topology)
    ag_chain = roles["antigen"]
    ab_chains = set(roles["antibody"])
    ag, ab = [], []
    for res in topology.residues():
        idxs = [a.index for a in res.atoms()]
        if res.chain.id == ag_chain:
            ag.extend(idxs)
        elif res.chain.id in ab_chains:
            ab.extend(idxs)
    if not ag or not ab:
        return float("nan")
    return float(np.linalg.norm(pos[ag].mean(0) - pos[ab].mean(0)))


def analyse_trajectory_mdtraj(out_dir: Path, topology_pdb: Path,
                               hotspot_seq: list[int]) -> dict:
    """
    MDTraj로 DCD 궤적을 분석합니다.

    분석 항목:
      - 백본 RMSD (최소화 구조 기준)
      - 잔기별 RMSF → B-factor 환산
      - PBC 보정 무게중심 거리 (항원 vs 항체)
      - 회전 반경 (Rg)
      - Hotspot 접촉 수 시계열

    Returns:
        분석 결과 딕셔너리
    """
    try:
        import mdtraj as md
    except ImportError:
        print("  [경고] MDTraj 미설치 — 궤적 분석 건너뜀. pip install mdtraj")
        return {}

    results: dict = {}

    for label in ("equil", "prod"):
        dcd = out_dir / f"{label}.dcd"
        if not dcd.exists():
            continue

        print(f"\n[MDTraj] {label} 궤적 분석...")
        try:
            traj = md.load(str(dcd), top=str(topology_pdb))
        except Exception as exc:
            print(f"  로드 실패: {exc}")
            continue

        # PBC 이미지 보정 (unwrapping)
        try:
            traj.image_molecules(inplace=True)
        except Exception:
            pass

        n_frames = traj.n_frames
        print(f"  프레임: {n_frames}  원자: {traj.n_atoms}")

        # ── 백본 RMSD (첫 프레임 기준) ───────────────────────────────────
        try:
            bb_idx = traj.topology.select("backbone and protein")
            if len(bb_idx) > 0:
                rmsd = md.rmsd(traj, traj, 0, atom_indices=bb_idx)
                results[f"{label}_rmsd_nm"] = {
                    "mean": float(rmsd.mean()),
                    "max":  float(rmsd.max()),
                    "final": float(rmsd[-1]),
                    "per_frame": rmsd.tolist(),
                }
                print(f"  백본 RMSD: mean={rmsd.mean()*10:.2f} Å  "
                      f"max={rmsd.max()*10:.2f} Å  final={rmsd[-1]*10:.2f} Å")
        except Exception as exc:
            print(f"  RMSD 계산 실패: {exc}")

        # ── RMSF → B-factor ──────────────────────────────────────────────
        try:
            ca_idx = traj.topology.select("name CA and protein")
            if len(ca_idx) > 0:
                rmsf = md.rmsf(traj, traj, 0, atom_indices=ca_idx)
                bfactor = (8 * np.pi**2 / 3) * (rmsf * 10)**2  # Å² → B-factor
                results[f"{label}_rmsf"] = {
                    "mean_A": float(rmsf.mean() * 10),
                    "max_A":  float(rmsf.max() * 10),
                    "per_ca_A": (rmsf * 10).tolist(),
                    "bfactor_per_ca": bfactor.tolist(),
                }
                print(f"  Cα RMSF: mean={rmsf.mean()*10:.2f} Å  max={rmsf.max()*10:.2f} Å")
        except Exception as exc:
            print(f"  RMSF 계산 실패: {exc}")

        # ── 무게중심 거리 (PBC 보정) ──────────────────────────────────────
        try:
            chains = list(traj.topology.chains)
            protein_chains = [c for c in chains
                              if any(r.name not in ("HOH", "WAT", "NA", "CL", "K")
                                     for r in c.residues)]
            if len(protein_chains) >= 2:
                ag_atoms = [a.index for a in protein_chains[0].atoms]
                ab_atoms = [a.index for r in protein_chains[1:]
                            for a in r.atoms]
                ag_centroid = traj.xyz[:, ag_atoms, :].mean(axis=1)
                ab_centroid = traj.xyz[:, ab_atoms, :].mean(axis=1)
                dist_nm = np.linalg.norm(ag_centroid - ab_centroid, axis=1)
                results[f"{label}_centroid_dist_nm"] = {
                    "mean":  float(dist_nm.mean()),
                    "min":   float(dist_nm.min()),
                    "final": float(dist_nm[-1]),
                    "per_frame": dist_nm.tolist(),
                }
                print(f"  무게중심 거리: mean={dist_nm.mean():.2f} nm  "
                      f"min={dist_nm.min():.2f} nm  final={dist_nm[-1]:.2f} nm")
        except Exception as exc:
            print(f"  무게중심 거리 계산 실패: {exc}")

        # ── 회전 반경 ─────────────────────────────────────────────────────
        try:
            prot_idx = traj.topology.select("protein")
            if len(prot_idx) > 0:
                rg = md.compute_rg(traj.atom_slice(prot_idx))
                results[f"{label}_rg_nm"] = {
                    "mean": float(rg.mean()),
                    "final": float(rg[-1]),
                }
                print(f"  회전 반경 Rg: mean={rg.mean()*10:.2f} Å  "
                      f"final={rg[-1]*10:.2f} Å")
        except Exception as exc:
            print(f"  Rg 계산 실패: {exc}")

        # ── Hotspot 접촉 시계열 ───────────────────────────────────────────
        try:
            chains_all = list(traj.topology.chains)
            protein_only = [c for c in chains_all
                            if any(r.name not in ("HOH", "WAT", "NA", "CL", "K")
                                   for r in c.residues)]
            if len(protein_only) >= 2:
                ag_chain = protein_only[0]
                ag_residues = list(ag_chain.residues)
                hotspot_res_indices = [r.index for r in ag_residues
                                       if (r.index + 1) in hotspot_seq]  # 0-based

                if hotspot_res_indices:
                    ab_res_indices = [r.index for c in protein_only[1:]
                                      for r in c.residues]
                    # 잔기 쌍 접촉 (cutoff 0.5 nm)
                    contact_pairs = [(hr, ar)
                                     for hr in hotspot_res_indices
                                     for ar in ab_res_indices[:50]]  # 상위 50개 제한
                    if contact_pairs:
                        dists, _ = md.compute_contacts(
                            traj, contact_pairs, scheme="closest-heavy"
                        )
                        n_contacts = (dists < 0.5).sum(axis=1)
                        results[f"{label}_hotspot_contacts"] = {
                            "mean": float(n_contacts.mean()),
                            "min":  float(n_contacts.min()),
                            "max":  float(n_contacts.max()),
                            "per_frame": n_contacts.tolist(),
                        }
                        print(f"  Hotspot 접촉 수: mean={n_contacts.mean():.1f}  "
                              f"min={n_contacts.min()}  max={n_contacts.max()}")
        except Exception as exc:
            print(f"  Hotspot 접촉 시계열 실패: {exc}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdb", type=Path,
                        default=REPO_ROOT / "design" / "docked_complex.pdb")
    parser.add_argument("--ff", choices=["ff14sb", "ff19sb"], default="ff14sb",
                        help="포스필드: ff14sb(기본, TIP3P) 또는 ff19sb(OPC, 정밀)")
    parser.add_argument("--steps", type=int, default=250000,
                        help="NpT 생산 MD 스텝 수 (0.002 ps; 기본 250000 = 500 ps)")
    parser.add_argument("--equil", type=int, default=50000,
                        help="NVT 평형화 스텝 수 (기본 50000 = 100 ps)")
    parser.add_argument("--salt", type=float, default=0.15,
                        help="NaCl 농도 mol/L (기본 0.15 M)")
    parser.add_argument("--temp", type=float, default=300.0,
                        help="온도 K (기본 300 K)")
    parser.add_argument("--dt", type=float, default=2.0,
                        help="타임스텝 fs (기본 2.0; HMR 적용 시 4.0 가능)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "openmm")
    parser.add_argument("--analysis_only", action="store_true",
                        help="기존 DCD 재분석만 수행 (MD 건너뜀)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fixed = args.out / "fixed.pdb"

    cfg = FF_CONFIGS.get(args.ff, FF_CONFIGS["ff14sb"])
    dt_ps = args.dt * 1e-3  # fs → ps

    print("=" * 60)
    print(f"포스필드: {cfg['label']}")
    print(f"타임스텝: {args.dt:.1f} fs")
    print(f"NVT 평형화: {args.equil} steps = {args.equil * dt_ps:.1f} ps")
    print(f"NpT 생산 MD: {args.steps} steps = {args.steps * dt_ps:.1f} ps")
    print(f"NaCl 농도: {args.salt} M  |  온도: {args.temp} K")
    print("=" * 60)

    # ── analysis_only 모드 ────────────────────────────────────────────────
    if args.analysis_only:
        topology_pdb = args.out / "minimized.pdb"
        if not topology_pdb.exists():
            topology_pdb = args.out / "fixed.pdb"
        print(f"[재분석] 기존 DCD 궤적 분석 ({topology_pdb.name} 기준)")
        hotspot_seq = list(range(1, 12))  # 대략적; 정확도 필요 시 --pdb 함께 사용
        md_results = analyse_trajectory_mdtraj(args.out, topology_pdb, hotspot_seq)
        with open(args.out / "md_analysis.json", "w") as f:
            json.dump(md_results, f, indent=2)
        print(f"분석 완료 → {args.out}/md_analysis.json")
        return

    # ── 0. Hotspot 매핑 (PDBFixer 전에 원본 번호로부터 계산) ─────────────
    print("\n[잔기 매핑] 원본 PDB에서 hotspot 순서 번호 계산...")
    hotspot_seq = map_hotspots_to_sequential(args.pdb, HOTSPOT_MATURE)
    print(f"  Sequential IDs: {hotspot_seq}")

    # ── 1. PDBFixer ──────────────────────────────────────────────────────
    fix_pdb(args.pdb, fixed)

    # ── 2. 시스템 구축 ────────────────────────────────────────────────────
    sim, modeller = build_system(
        fixed, ff_key=args.ff, padding=1.0,
        salt_molar=args.salt, temperature_K=args.temp, dt_fs=args.dt,
    )

    # ── 3. 단계적 구속 완화 최소화 ─────────────────────────────────────────
    # LightDock rigid-body pose has severe clashes → staged Cα restraints
    # k schedule: 1000 → 100 → 10 → 0 kJ/mol/nm²
    print("\n[구속 최소화] Cα 구속 하에 단계적 에너지 최소화")
    restraint = add_ca_restraints(sim, modeller, k_kj=1000.0)

    e_min = None
    for k_val in [1000.0, 100.0, 10.0, 0.0]:
        update_restraint_k(sim, restraint, k_val)
        print(f"\n  k = {k_val:.0f} kJ/mol/nm²")
        e_min = minimize(sim, tol=10.0, label=f"k{int(k_val)}")

    # 최소화된 구조 저장
    min_pdb = args.out / "minimized.pdb"
    state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(min_pdb, "w") as f:
        PDBFile.writeFile(modeller.topology, state.getPositions(), f)
    print(f"  최소화 구조: {min_pdb}")

    # 복합체 분리 확인
    d_cen = centroid_distance(sim, modeller.topology)
    print(f"  항원-항체 무게중심 거리: {d_cen:.2f} nm")

    # 최소화 후 접촉 분석
    contacts_min = analyse_contacts(sim, modeller.topology, hotspot_seq)
    if contacts_min["min_dist_nm"] is not None:
        print(f"\n  Hotspot 접촉 (최소화 후): {contacts_min['hotspot_contacts']}/"
              f"{contacts_min['total_hotspot_atoms']} atoms  "
              f"최단거리 {contacts_min['min_dist_nm']:.3f} nm")
    else:
        print("\n  Hotspot 접촉 (최소화 후): 체인 미탐지 (체인 재매핑 확인 필요)")

    # ── 4. NVT 평형화 ─────────────────────────────────────────────────────
    if args.equil > 0:
        sim.context.setVelocitiesToTemperature(args.temp * unit.kelvin)
        run_md(sim, args.out, args.equil, label="equil")

    # ── 5. NpT 생산 MD ────────────────────────────────────────────────────
    if args.steps > 0:
        run_md(sim, args.out, args.steps, label="prod")

    # 최종 접촉 분석
    contacts_prod = analyse_contacts(sim, modeller.topology, hotspot_seq)
    if contacts_prod["min_dist_nm"] is not None:
        print(f"\n  Hotspot 접촉 (MD 후): {contacts_prod['hotspot_contacts']}/"
              f"{contacts_prod['total_hotspot_atoms']} atoms  "
              f"최단거리 {contacts_prod['min_dist_nm']:.3f} nm")
    else:
        print("\n  Hotspot 접촉 (MD 후): 체인 미탐지")

    # 최종 구조 저장
    prod_pdb = args.out / "production.pdb"
    state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(prod_pdb, "w") as f:
        PDBFile.writeFile(modeller.topology, state.getPositions(), f)
    print(f"  생산 MD 구조: {prod_pdb}")

    # ── 6. MDTraj 궤적 분석 ───────────────────────────────────────────────
    print("\n[MDTraj] 궤적 분석 시작...")
    md_results = analyse_trajectory_mdtraj(args.out, min_pdb, hotspot_seq)
    with open(args.out / "md_analysis.json", "w") as f:
        json.dump(md_results, f, indent=2)

    # ── 7. 결과 저장 ──────────────────────────────────────────────────────
    result = {
        "input_pdb": str(args.pdb),
        "forcefield": cfg["label"],
        "ff_key": args.ff,
        "water_model": cfg["water_model"],
        "salt_M": args.salt,
        "temperature_K": args.temp,
        "timestep_fs": args.dt,
        "minimization": {
            "final_energy_kJ_mol": round(e_min, 1) if e_min is not None else None,
            "centroid_distance_nm": round(d_cen, 3),
            "restraint_schedule_kJ_mol_nm2": [1000, 100, 10, 0],
        },
        "equil_steps": args.equil,
        "equil_time_ps": round(args.equil * dt_ps, 2),
        "prod_steps": args.steps,
        "prod_time_ps": round(args.steps * dt_ps, 2),
        "hotspot_contacts_after_min": contacts_min,
        "hotspot_contacts_after_md": contacts_prod,
        "mdtraj_analysis": md_results,
    }
    (args.out / "validation_result.json").write_text(json.dumps(result, indent=2))

    # ── 8. 요약 출력 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OpenMM MD 검증 완료")
    print(f"  포스필드:      {cfg['label']}")
    print(f"  에너지 최소화: {e_min:,.0f} kJ/mol" if e_min else "  에너지 최소화: N/A")
    print(f"  무게중심 거리: {d_cen:.2f} nm (복합체 {'유지 ✓' if d_cen < 3.0 else '분리됨!'})")
    prod_t = args.steps * dt_ps
    equil_t = args.equil * dt_ps
    print(f"  MD 시간:       평형화 {equil_t:.0f} ps + 생산 {prod_t:.0f} ps")
    if contacts_prod["min_dist_nm"] is not None:
        print(f"  Hotspot 접촉:  {contacts_prod['hotspot_contacts']} / "
              f"{contacts_prod['total_hotspot_atoms']} (MD 후)")
    # MDTraj 요약
    rmsd_key = "prod_rmsd_nm"
    if rmsd_key in md_results:
        rmsd = md_results[rmsd_key]
        print(f"  백본 RMSD:     mean={rmsd['mean']*10:.2f} Å  "
              f"max={rmsd['max']*10:.2f} Å")
    rmsf_key = "prod_rmsf"
    if rmsf_key in md_results:
        rmsf = md_results[rmsf_key]
        print(f"  Cα RMSF:       mean={rmsf['mean_A']:.2f} Å  max={rmsf['max_A']:.2f} Å")
    print(f"  결과:          {args.out}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
