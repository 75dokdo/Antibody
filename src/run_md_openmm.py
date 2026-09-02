#!/usr/bin/env python3
"""
OpenMM MD 시뮬레이션 - FAP ECD + scFv 복합체
ff19SB 힘 장 + OPC 수모델, M5 Pro Metal GPU

입력: fap_design/colabfold/results/<id>_relaxed_rank_001_*.pdb
출력: fap_design/md/<id>/  (trajectory.dcd, final.pdb, energies.csv)

의존성:
  conda install -c conda-forge openmm openmmforcefields pdbfixer
  또는
  pip install openmm openmmforcefields pdbfixer

사용법:
  python3 src/run_md_openmm.py --pdb <input.pdb> --out_dir fap_design/md/<id>
  python3 src/run_md_openmm.py --batch  # Top5 모두 실행
"""

import argparse
import os
import sys
import json
import time


def fix_pdb(pdb_path: str, fixed_path: str):
    """pdbfixer로 PDB 정제 (결측 잔기/원자 추가, 수소 제거)."""
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)  # pH 7.4

    with open(fixed_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    print(f"  [pdbfixer] → {fixed_path}")


def run_md(pdb_path: str, out_dir: str, steps: int = 50_000_000,
           temperature_K: float = 310.0, report_interval: int = 10_000,
           platform_name: str = "Metal"):
    """
    OpenMM MD 실행.
    steps=50M × 2fs = 100 ns
    report_interval=10k → 10k×2fs = 20 ps 간격 (5000 frames)
    """
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from openmmforcefields.generators import SystemGenerator

    os.makedirs(out_dir, exist_ok=True)

    # ── 플랫폼 선택 ──
    try:
        platform = mm.Platform.getPlatformByName(platform_name)
        print(f"  [OpenMM] 플랫폼: {platform_name}")
    except Exception:
        print(f"  [OpenMM] {platform_name} 불가 → CPU 사용")
        platform = mm.Platform.getPlatformByName("CPU")

    # ── PDB 로드 ──
    pdb = app.PDBFile(pdb_path)

    # ── 힘 장 생성 (ff19SB + OPC) ──
    forcefield = app.ForceField("amber/ff19SB.xml", "amber/opc.xml")

    # ── 시스템 생성 (PBC, PME, HBonds) ──
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(
        forcefield,
        model="opc",
        padding=1.0 * unit.nanometer,
        positiveIon="Na+",
        negativeIon="Cl-",
        ionicStrength=0.15 * unit.molar,
    )

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        hydrogenMass=1.5 * unit.amu,  # HMR: 4fs timestep 가능
    )

    # ── 온도/압력 배스 ──
    integrator = mm.LangevinMiddleIntegrator(
        temperature_K * unit.kelvin,
        1.0 / unit.picosecond,  # collision frequency
        0.002 * unit.picoseconds,  # timestep 2fs (HMR 사용 시 4fs 가능)
    )
    system.addForce(mm.MonteCarloBarostat(
        1.0 * unit.bar,
        temperature_K * unit.kelvin,
        25,  # frequency
    ))

    simulation = app.Simulation(
        modeller.topology, system, integrator, platform
    )
    simulation.context.setPositions(modeller.positions)

    # ── 에너지 최소화 ──
    print("  [MD] 에너지 최소화 시작...")
    t0 = time.time()
    simulation.minimizeEnergy(maxIterations=2000)
    print(f"  [MD] 최소화 완료 ({time.time()-t0:.1f}s)")

    # ── NVT 평형 (50 ps) ──
    print("  [MD] NVT 평형 (50 ps)...")
    simulation.context.setVelocitiesToTemperature(temperature_K * unit.kelvin)
    simulation.step(25_000)  # 25k × 2fs = 50 ps

    # ── NPT 평형 (500 ps) ──
    print("  [MD] NPT 평형 (500 ps)...")
    simulation.step(250_000)  # 250k × 2fs = 500 ps

    # ── 생산 MD (100 ns) ──
    traj_path = os.path.join(out_dir, "trajectory.dcd")
    energy_path = os.path.join(out_dir, "energies.csv")
    final_pdb_path = os.path.join(out_dir, "final.pdb")
    checkpoint_path = os.path.join(out_dir, "checkpoint.chk")

    simulation.reporters.append(
        app.DCDReporter(traj_path, report_interval)
    )
    simulation.reporters.append(
        app.StateDataReporter(
            energy_path,
            report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            temperature=True,
            volume=True,
            separator=",",
        )
    )
    simulation.reporters.append(
        app.CheckpointReporter(checkpoint_path, 1_000_000)  # checkpoint 2ns마다
    )

    print(f"  [MD] 생산 MD 시작 ({steps*0.002/1000:.0f} ns = {steps:,} steps)...")
    t_prod = time.time()
    simulation.step(steps)
    elapsed = time.time() - t_prod
    print(f"  [MD] 완료! 소요: {elapsed/3600:.2f}h")

    # ── 최종 구조 저장 ──
    positions = simulation.context.getState(getPositions=True,
                                            enforcePeriodicBox=True).getPositions()
    with open(final_pdb_path, "w") as f:
        app.PDBFile.writeFile(simulation.topology, positions, f)
    print(f"  [MD] 최종 구조: {final_pdb_path}")
    print(f"  [MD] 궤적: {traj_path}")

    return {
        "out_dir": out_dir,
        "steps": steps,
        "ns": steps * 0.002 / 1000,
        "elapsed_h": round(elapsed / 3600, 2),
        "trajectory": traj_path,
        "final_pdb": final_pdb_path,
        "energies": energy_path,
    }


def parse_args():
    p = argparse.ArgumentParser(description="OpenMM MD for FAP scFv complex")
    p.add_argument("--pdb", default=None, help="입력 PDB 경로")
    p.add_argument("--out_dir", default="fap_design/md/default",
                   help="출력 디렉토리")
    p.add_argument("--platform", default="Metal",
                   choices=["Metal", "CUDA", "OpenCL", "CPU"],
                   help="OpenMM 플랫폼 (M5 Pro = Metal)")
    p.add_argument("--ns", type=float, default=100.0,
                   help="시뮬레이션 길이 (ns, default 100)")
    p.add_argument("--temp", type=float, default=310.0,
                   help="온도 (K, default 310 = 37°C)")
    p.add_argument("--batch", action="store_true",
                   help="Top5 모두 실행 (ColabFold best model 자동 탐색)")
    p.add_argument("--top_n", type=int, default=3,
                   help="배치 시 상위 N개만 실행 (default 3)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.batch:
        # ColabFold 결과에서 Top N 후보 PDB 탐색
        summary_path = "fap_design/colabfold/complex_summary.json"
        if not os.path.exists(summary_path):
            print(f"[ERROR] ColabFold 결과 없음: {summary_path}")
            print("먼저: python3 src/analyze_colabfold.py 실행")
            sys.exit(1)

        with open(summary_path) as f:
            summary = json.load(f)

        ranked = sorted(summary,
                        key=lambda x: x.get("iptm") or 0,
                        reverse=True)[:args.top_n]

        for rank, cand in enumerate(ranked, 1):
            cid = cand["id"]
            best_pdb = cand.get("best_pdb")
            if not best_pdb or not os.path.exists(best_pdb):
                print(f"[{rank}/{args.top_n}] {cid}: PDB 없음, 건너뜀")
                continue

            out_dir = f"fap_design/md/{cid}"
            print(f"\n[{rank}/{args.top_n}] {cid} (ipTM={cand.get('iptm','N/A')})")

            # pdbfixer
            fixed_pdb = os.path.join(out_dir, "input_fixed.pdb")
            os.makedirs(out_dir, exist_ok=True)
            try:
                fix_pdb(best_pdb, fixed_pdb)
            except ImportError:
                print("  [경고] pdbfixer 미설치 → 원본 PDB 직접 사용")
                fixed_pdb = best_pdb

            steps = int(args.ns * 1000 / 0.002)  # ns → steps (2fs timestep)
            result = run_md(fixed_pdb, out_dir, steps=steps,
                            temperature_K=args.temp,
                            platform_name=args.platform)

            result_path = os.path.join(out_dir, "md_result.json")
            with open(result_path, "w") as f:
                json.dump({**cand, **result}, f, indent=2)
            print(f"  [결과] {result_path}")

    else:
        # 단일 PDB 실행
        if not args.pdb:
            print("[ERROR] --pdb 또는 --batch 필요")
            sys.exit(1)

        if not os.path.exists(args.pdb):
            print(f"[ERROR] PDB 없음: {args.pdb}")
            sys.exit(1)

        cid = os.path.basename(args.pdb).replace(".pdb", "")
        fixed_pdb = os.path.join(args.out_dir, "input_fixed.pdb")
        os.makedirs(args.out_dir, exist_ok=True)

        try:
            fix_pdb(args.pdb, fixed_pdb)
        except ImportError:
            print("[경고] pdbfixer 미설치 → 원본 PDB 사용")
            fixed_pdb = args.pdb

        steps = int(args.ns * 1000 / 0.002)
        result = run_md(fixed_pdb, args.out_dir, steps=steps,
                        temperature_K=args.temp,
                        platform_name=args.platform)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
