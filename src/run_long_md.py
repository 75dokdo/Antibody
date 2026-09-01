#!/usr/bin/env python3
"""
run_long_md.py — 100–200 ns 생산 MD (ff19SB + OPC, HMR)

openmm_validate.py로 에너지 최소화·단기 평형화를 완료한 뒤,
장기 생산 MD를 돌리는 전용 스크립트입니다.

핵심 최적화:
  · HMR (Hydrogen Mass Repartitioning) → dt = 4 fs (2× 가속)
  · 단백질 전용 DCD — 물/이온 제외, 프레임당 52 KB (10× 절감)
  · 10 ps/frame 저장 → 100 ns = 10,000 frames = ~500 MB
  · 1 ns마다 체크포인트 → 중단 후 재시작 가능
  · GPU 필수 (CPU는 100 ns에 100일 소요)

소요 시간 추정 (44,000 원자 솔베이션, GPU):
  RTX 3060 Ti: 100 ns ≈ 12시간  |  200 ns ≈ 1일
  M5 Mac Metal: 100 ns ≈ 1.7일  |  200 ns ≈ 3.3일

사용법:
    # 처음 실행 (openmm_validate.py --ff ff19sb 이후)
    python src/run_long_md.py \\
        --minimized results/openmm/minimized.pdb \\
        --out results/long_md \\
        --ns 100

    # 체크포인트에서 재시작
    python src/run_long_md.py \\
        --minimized results/openmm/minimized.pdb \\
        --out results/long_md \\
        --ns 100 --resume

    # HMR 없이 (2 fs, 느리지만 보수적)
    python src/run_long_md.py \\
        --minimized results/openmm/minimized.pdb \\
        --out results/long_md \\
        --ns 100 --no_hmr

출력:
    results/long_md/
        equil_long.dcd        — 10 ns NVT 평형화 (단백질 전용)
        prod.dcd              — 생산 MD (단백질 전용)
        prod.csv              — 에너지/온도/밀도 시계열
        checkpoint.chk        — 최신 체크포인트
        checkpoints/          — 1 ns마다 백업 체크포인트
        md_analysis.json      — MDTraj RMSD/RMSF/접촉 분석
        run_info.json         — 실행 설정 + 성능 통계

설치:
    pip install openmm pdbfixer mdtraj scipy
    # GPU (CUDA or Metal) 필수
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# ── 프로젝트 임포트 ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from openmm_validate import (
    FF_CONFIGS,
    HOTSPOT_MATURE,
    analyse_contacts,
    analyse_trajectory_mdtraj,
    centroid_distance,
)

# ── 상수 ──────────────────────────────────────────────────────────────────────
# 10 ps마다 저장 → 100 ns = 10,000 frames ≈ 500 MB (단백질 전용)
DEFAULT_SAVE_INTERVAL_PS = 10.0
# 1 ns마다 체크포인트
DEFAULT_CHECKPOINT_INTERVAL_NS = 1.0
# 기본 HMR 수소 질량 (3× → dt=4 fs 안전)
HMR_MASS_AMU = 3.0


def estimate_storage(n_prot_atoms: int, n_frames: int, n_frames_equil: int) -> dict:
    """저장 용량 사전 추정."""
    frame_bytes = n_prot_atoms * 3 * 4  # float32, 3D
    prod_mb  = frame_bytes * n_frames / 1e6
    equil_mb = frame_bytes * n_frames_equil / 1e6
    return {
        "frame_kb": frame_bytes / 1024,
        "prod_dcd_mb": prod_mb,
        "equil_dcd_mb": equil_mb,
        "total_mb": prod_mb + equil_mb + 10,  # +10 MB misc
    }


def build_long_md_system(
    minimized_pdb: Path,
    ff_key: str = "ff19sb",
    temperature_K: float = 300.0,
    dt_fs: float = 4.0,
    hmr: bool = True,
    salt_molar: float = 0.15,
):
    """
    최소화된 PDB에서 장기 MD 시스템을 구성합니다.

    openmm_validate.py의 build_system()과 달리 addSolvent를 다시 하지 않습니다.
    minimized.pdb(물 포함)을 그대로 읽어 포스필드만 적용합니다.

    HMR: 수소 원자 질량을 3× 증가 → 제약 없이 dt=4 fs 사용 가능.
    """
    from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, unit
    from openmm.app import (
        ForceField, HBonds, Modeller, PDBFile, PME, Simulation,
    )

    cfg = FF_CONFIGS.get(ff_key, FF_CONFIGS["ff19sb"])
    print(f"[시스템] {cfg['label']}  dt={dt_fs:.1f} fs  HMR={hmr}")

    pdb = PDBFile(str(minimized_pdb))
    ff  = ForceField(*cfg["ff_files"])

    create_kwargs = dict(
        nonbondedMethod=PME,
        nonbondedCutoff=cfg["cutoff_nm"] * unit.nanometers,
        constraints=HBonds,
        rigidWater=True,
        ewaldErrorTolerance=5e-4,
    )
    if hmr:
        create_kwargs["hydrogenMass"] = HMR_MASS_AMU * unit.amu
        print(f"  HMR 활성: H 질량 {HMR_MASS_AMU}× → dt={dt_fs:.1f} fs 안전")

    system = ff.createSystem(pdb.topology, **create_kwargs)
    system.addForce(MonteCarloBarostat(1 * unit.bar, temperature_K * unit.kelvin))

    integrator = LangevinMiddleIntegrator(
        temperature_K * unit.kelvin,
        1.0 / unit.picosecond,
        dt_fs * unit.femtoseconds,
    )

    # GPU 자동 선택
    platform_name = "CPU"
    props = {}
    for pname, p in [("CUDA", {"CudaPrecision": "mixed"}),
                     ("OpenCL", {}), ("CPU", {})]:
        try:
            plat = Platform.getPlatformByName(pname)
            platform_name = pname
            props = p
            break
        except Exception:
            continue

    sim = Simulation(pdb.topology, system, integrator,
                     Platform.getPlatformByName(platform_name), props)
    sim.context.setPositions(pdb.positions)

    # 박스 벡터 복원
    if pdb.topology.getPeriodicBoxVectors() is not None:
        sim.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    print(f"  플랫폼: {platform_name}")
    return sim, pdb.topology


def get_protein_atom_indices(topology) -> list[int]:
    """물/이온을 제외한 단백질 원자 인덱스 목록."""
    water_names = {"HOH", "WAT", "CL", "NA", "K", "MG", "CA", "ZN"}
    return [a.index for a in topology.atoms()
            if a.residue.name not in water_names]


def run_production(
    sim,
    topology,
    out_dir: Path,
    total_steps: int,
    dt_fs: float,
    save_interval_steps: int,
    checkpoint_interval_steps: int,
    label: str = "prod",
    resume: bool = False,
) -> dict:
    """
    체크포인트 지원 장기 생산 MD.

    Returns:
        {'n_steps_done': int, 'elapsed_s': float, 'ns_per_day': float}
    """
    from openmm import unit
    from openmm.app import DCDReporter, StateDataReporter, CheckpointReporter

    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.chk"

    # 단백질 전용 원자 인덱스
    prot_idx = get_protein_atom_indices(topology)
    n_prot   = len(prot_idx)
    n_all    = topology.getNumAtoms()
    print(f"  단백질 원자: {n_prot:,} / 전체: {n_all:,}  "
          f"(DCD 절감: {(1 - n_prot/n_all)*100:.0f}%)")

    # 재시작
    steps_done = 0
    if resume and checkpoint_path.exists():
        sim.loadCheckpoint(str(checkpoint_path))
        # 현재 스텝 추정 (StateData에서 읽기 어려우므로 사용자 알림)
        print(f"  체크포인트 로드: {checkpoint_path}")
        print("  주의: 재시작 스텝 수는 run_info.json에서 확인")
    else:
        sim.context.setVelocitiesToTemperature(300 * unit.kelvin)

    # 리포터 설정
    sim.reporters.clear()
    dcd_path = out_dir / f"{label}.dcd"
    sim.reporters.append(DCDReporter(
        str(dcd_path), save_interval_steps,
        atomSubset=prot_idx,
        append=resume and dcd_path.exists(),
    ))
    sim.reporters.append(StateDataReporter(
        str(out_dir / f"{label}.csv"), save_interval_steps,
        step=True, time=True, potentialEnergy=True,
        kineticEnergy=True, temperature=True, density=True,
        append=resume and (out_dir / f"{label}.csv").exists(),
    ))
    sim.reporters.append(CheckpointReporter(
        str(checkpoint_path), checkpoint_interval_steps,
    ))
    # 진행률: 5번 출력
    sim.reporters.append(StateDataReporter(
        None, max(1, total_steps // 5),
        step=True, time=True, potentialEnergy=True, temperature=True,
        progress=True, remainingTime=True, totalSteps=total_steps,
        separator="\t",
    ))

    total_ns = total_steps * dt_fs * 1e-6
    est_frames = total_steps // save_interval_steps
    est_mb = n_prot * 3 * 4 * est_frames / 1e6
    print(f"\n[생산 MD] {total_steps:,} steps = {total_ns:.0f} ns  "
          f"({est_frames:,} frames, ~{est_mb:.0f} MB)")

    t0 = time.time()

    # 1 ns 블록 단위로 실행 (체크포인트 백업)
    block_steps = checkpoint_interval_steps
    n_blocks = (total_steps + block_steps - 1) // block_steps
    completed_steps = 0

    for blk in range(n_blocks):
        remaining = total_steps - completed_steps
        this_block = min(block_steps, remaining)
        if this_block <= 0:
            break

        sim.step(this_block)
        completed_steps += this_block

        # 체크포인트 백업 (1 ns마다)
        elapsed_ns = completed_steps * dt_fs * 1e-6
        bkp = checkpoint_dir / f"chk_{elapsed_ns:.0f}ns.chk"
        if checkpoint_path.exists():
            shutil.copy(checkpoint_path, bkp)

        elapsed_s = time.time() - t0
        ns_per_day = (elapsed_ns / elapsed_s) * 86400 if elapsed_s > 0 else 0
        remaining_ns = (total_steps - completed_steps) * dt_fs * 1e-6
        eta_h = remaining_ns / (ns_per_day / 24) if ns_per_day > 0 else 0
        print(f"  [{blk+1}/{n_blocks}] {elapsed_ns:.0f}/{total_ns:.0f} ns  "
              f"{ns_per_day:.1f} ns/day  ETA {eta_h:.1f}h")

    elapsed_s = time.time() - t0
    ns_per_day = (total_ns / elapsed_s) * 86400 if elapsed_s > 0 else 0
    print(f"  완료 [{elapsed_s:.0f}s = {elapsed_s/3600:.2f}h]  "
          f"성능: {ns_per_day:.1f} ns/day")

    return {
        "n_steps_done": completed_steps,
        "elapsed_s": elapsed_s,
        "ns_per_day": ns_per_day,
        "dcd_mb": est_mb,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minimized", type=Path,
                    default=REPO_ROOT / "results" / "openmm" / "minimized.pdb",
                    help="openmm_validate.py가 생성한 최소화 PDB (물 포함)")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "long_md")
    ap.add_argument("--ff", choices=["ff14sb", "ff19sb"], default="ff19sb",
                    help="포스필드 (기본: ff19sb+OPC — HMR과 조합 최적)")
    ap.add_argument("--ns", type=float, default=100.0,
                    help="생산 MD 길이 (ns, 기본 100)")
    ap.add_argument("--equil_ns", type=float, default=10.0,
                    help="추가 NVT 평형화 (ns, 기본 10)")
    ap.add_argument("--no_hmr", action="store_true",
                    help="HMR 비활성화 → dt=2 fs (느리지만 보수적)")
    ap.add_argument("--dt", type=float, default=None,
                    help="타임스텝 fs (기본: HMR=4, no-HMR=2)")
    ap.add_argument("--save_ps", type=float, default=DEFAULT_SAVE_INTERVAL_PS,
                    help="DCD/CSV 저장 간격 ps (기본 10 ps)")
    ap.add_argument("--checkpoint_ns", type=float, default=DEFAULT_CHECKPOINT_INTERVAL_NS,
                    help="체크포인트 간격 ns (기본 1 ns)")
    ap.add_argument("--temp", type=float, default=300.0,
                    help="온도 K (기본 300)")
    ap.add_argument("--resume", action="store_true",
                    help="checkpoint.chk에서 재시작")
    ap.add_argument("--analysis_only", action="store_true",
                    help="기존 DCD 재분석만 (MD 건너뜀)")
    ap.add_argument("--hotspot_seq", type=int, nargs="+",
                    default=[13, 16, 19, 20, 23, 32, 33, 35, 36, 39, 40],
                    help="GPC3 hotspot sequential IDs (기본: 프로젝트 매핑값)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # 타임스텝 결정
    hmr = not args.no_hmr
    if args.dt is not None:
        dt_fs = args.dt
    else:
        dt_fs = 4.0 if hmr else 2.0
    dt_ps = dt_fs * 1e-3

    # 스텝 수 계산
    prod_steps   = int(args.ns * 1e3 / dt_ps)
    equil_steps  = int(args.equil_ns * 1e3 / dt_ps)
    save_steps   = int(args.save_ps / dt_ps)
    chkpt_steps  = int(args.checkpoint_ns * 1e3 / dt_ps)

    # 사전 정보 출력
    print("=" * 65)
    print(f"장기 MD 설정")
    print(f"  포스필드:  {FF_CONFIGS[args.ff]['label']}")
    print(f"  HMR:       {'활성 (dt=4 fs)' if hmr else '비활성 (dt=2 fs)'}")
    print(f"  타임스텝:  {dt_fs:.1f} fs")
    print(f"  평형화:    {args.equil_ns:.0f} ns  ({equil_steps:,} steps)")
    print(f"  생산 MD:   {args.ns:.0f} ns  ({prod_steps:,} steps)")
    print(f"  DCD 저장:  매 {args.save_ps:.0f} ps  ({save_steps} steps/frame)")
    print(f"  체크포인트: 매 {args.checkpoint_ns:.0f} ns")
    print(f"  온도:      {args.temp} K")

    # 저장 용량 사전 추정 (단백질 원자 수 추정: 고정값)
    n_prot_est = 4342  # 실측; 다른 복합체면 실제 로드 후 갱신됨
    prod_frames  = prod_steps // save_steps
    equil_frames = equil_steps // save_steps
    est = estimate_storage(n_prot_est, prod_frames, equil_frames)
    print(f"\n  저장 용량 추정 (단백질 전용 DCD):")
    print(f"    프레임당: {est['frame_kb']:.0f} KB")
    print(f"    equil.dcd: {est['equil_dcd_mb']:.0f} MB  "
          f"({equil_frames:,} frames)")
    print(f"    prod.dcd:  {est['prod_dcd_mb']:.0f} MB  "
          f"({prod_frames:,} frames)")
    print(f"    합계:      {est['total_mb']:.0f} MB  "
          f"(≈ {est['total_mb']/1024:.1f} GB)")
    print("=" * 65)

    if not args.minimized.exists():
        print(f"\n[오류] minimized.pdb 없음: {args.minimized}")
        print("  먼저 실행: python src/openmm_validate.py --ff ff19sb")
        sys.exit(1)

    # ── analysis_only 모드 ─────────────────────────────────────────────────
    if args.analysis_only:
        print("\n[재분석] 기존 DCD 분석")
        results = analyse_trajectory_mdtraj(
            args.out, args.minimized, args.hotspot_seq)
        with open(args.out / "md_analysis.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"완료 → {args.out}/md_analysis.json")
        return

    # ── 시스템 구축 ───────────────────────────────────────────────────────
    sim, topology = build_long_md_system(
        args.minimized, ff_key=args.ff,
        temperature_K=args.temp, dt_fs=dt_fs, hmr=hmr,
    )

    # 실제 단백질 원자 수 업데이트
    prot_idx = get_protein_atom_indices(topology)
    n_prot = len(prot_idx)
    print(f"\n  실제 단백질 원자: {n_prot:,}개")
    est = estimate_storage(n_prot, prod_frames, equil_frames)
    print(f"  실제 저장 추정: prod {est['prod_dcd_mb']:.0f} MB  "
          f"총 {est['total_mb']:.0f} MB")

    # ── NVT 추가 평형화 ───────────────────────────────────────────────────
    if equil_steps > 0 and not args.resume:
        print(f"\n[평형화] {args.equil_ns:.0f} ns NVT...")
        from openmm.app import DCDReporter, StateDataReporter
        from openmm import unit

        sim.context.setVelocitiesToTemperature(args.temp * unit.kelvin)
        sim.reporters.clear()
        sim.reporters.append(DCDReporter(
            str(args.out / "equil_long.dcd"), save_steps,
            atomSubset=prot_idx,
        ))
        sim.reporters.append(StateDataReporter(
            str(args.out / "equil_long.csv"), save_steps,
            step=True, time=True, temperature=True, potentialEnergy=True,
        ))
        sim.reporters.append(StateDataReporter(
            None, max(1, equil_steps // 5),
            step=True, time=True, temperature=True, progress=True,
            remainingTime=True, totalSteps=equil_steps, separator="\t",
        ))
        t0 = time.time()
        sim.step(equil_steps)
        print(f"  평형화 완료 [{time.time()-t0:.0f}s]")

    # ── 생산 MD ───────────────────────────────────────────────────────────
    prod_stats = run_production(
        sim, topology, args.out,
        total_steps=prod_steps,
        dt_fs=dt_fs,
        save_interval_steps=save_steps,
        checkpoint_interval_steps=chkpt_steps,
        label="prod",
        resume=args.resume,
    )

    # ── 최종 구조 저장 ────────────────────────────────────────────────────
    from openmm import unit
    from openmm.app import PDBFile
    state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(args.out / "final.pdb", "w") as f:
        PDBFile.writeFile(topology, state.getPositions(), f)
    print(f"  최종 구조: {args.out}/final.pdb")

    # ── MDTraj 분석 ───────────────────────────────────────────────────────
    print("\n[MDTraj] 궤적 분석...")
    md_results = analyse_trajectory_mdtraj(
        args.out, args.minimized, args.hotspot_seq)
    with open(args.out / "md_analysis.json", "w") as f:
        json.dump(md_results, f, indent=2)

    # ── 실행 정보 저장 ────────────────────────────────────────────────────
    run_info = {
        "forcefield": FF_CONFIGS[args.ff]["label"],
        "hmr": hmr,
        "dt_fs": dt_fs,
        "prod_ns": args.ns,
        "equil_ns": args.equil_ns,
        "temperature_K": args.temp,
        "save_interval_ps": args.save_ps,
        "checkpoint_interval_ns": args.checkpoint_ns,
        "prod_steps_done": prod_stats["n_steps_done"],
        "elapsed_h": prod_stats["elapsed_s"] / 3600,
        "performance_ns_per_day": prod_stats["ns_per_day"],
        "prod_dcd_mb": prod_stats["dcd_mb"],
        "mdtraj_summary": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_frame"}
            for k, v in md_results.items()
            if isinstance(v, dict)
        },
    }
    with open(args.out / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    # ── 최종 요약 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("장기 MD 완료")
    print(f"  포스필드:  {FF_CONFIGS[args.ff]['label']}")
    print(f"  HMR:       {'활성 (4 fs)' if hmr else '비활성 (2 fs)'}")
    print(f"  시뮬레이션: {args.ns:.0f} ns")
    print(f"  성능:      {prod_stats['ns_per_day']:.1f} ns/day")
    print(f"  소요시간:  {prod_stats['elapsed_s']/3600:.2f}시간")
    print(f"  저장용량:  {prod_stats['dcd_mb']:.0f} MB")
    if "prod_rmsd_nm" in md_results:
        r = md_results["prod_rmsd_nm"]
        print(f"  백본 RMSD: mean={r['mean']*10:.2f} Å  max={r['max']*10:.2f} Å")
    print(f"  출력:      {args.out}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
