#!/bin/bash
# M5 Pro (Apple Silicon arm64) 항체 설계 도구 설치 스크립트
# macOS 15+ (Sequoia) 권장
# 설치 도구: OpenMM, ESM-2, ColabFold, Boltz-2, IgFold, AntiFold, AbLang2,
#            ProteinMPNN, ANARCI, MDTraj, PDBFixer 등

set -e
echo "=== M5 Pro 항체 설계 환경 설치 ==="

# 1. Miniforge (arm64 Conda) 설치 확인
if ! command -v conda &>/dev/null; then
    echo "[1/9] Miniforge 설치 중..."
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh \
        -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
    conda init zsh
    echo ""
    echo ">>> Miniforge 설치 완료. 터미널을 재시작한 후 이 스크립트를 다시 실행하세요."
    exit 0
fi

# 2. Conda 환경 생성 (Python 3.11)
echo "[2/9] Conda 환경 'antibody' 생성 (Python 3.11)..."
conda create -n antibody python=3.11 -y 2>/dev/null || echo "  → 이미 존재, 계속 진행"
eval "$(conda shell.bash hook)"
conda activate antibody

# 3. OpenMM (Metal GPU) + ff19SB/OPC 포함 + 구조 분석
echo "[3/9] OpenMM (Metal GPU) + ff19SB/OPC + MDTraj + PDBFixer + MDAnalysis..."
conda install -c conda-forge openmm openmmforcefields pdbfixer mdtraj mdanalysis -y
# openmmforcefields: ff14SB, ff19SB, OPC, GAFF2 제공 (boltz 등과 pip 충돌 없음)

# 4. PyTorch (MPS Metal 백엔드)
# Apple Silicon arm64용 - CUDA 없이 MPS로 GPU 가속
echo "[4/9] PyTorch (MPS Metal 지원)..."
pip install torch torchvision torchaudio

# 5. 항체 언어모델 / 스코어링
echo "[5/9] 항체 언어모델 (ESM-2, AbLang2, ANARCI)..."
pip install fair-esm                     # ESM-2 masked marginal PLL
pip install ablang2                      # 항체 전용 언어모델
pip install anarci                       # Chothia/IMGT 넘버링
pip install abnumber                     # 경량 항체 넘버링
pip install biopython==1.84

# 6. 항체 구조 예측
echo "[6/9] 항체 구조 예측 (IgFold, AntiFold, ColabFold, Boltz-2)..."
# IgFold — 항체 특화 빠른 구조 예측 (~1분/후보)
pip install igfold

# AntiFold — 항체 역폴딩 (구조→서열 설계, ESM-IF1 기반)
pip install antifold

# ColabFold — AlphaFold2-Multimer CPU 모드
pip install "colabfold[cpu]"

# Boltz-2 — MPS 지원 확산 모델 구조 예측
pip install boltz

# 7. ProteinMPNN (서열 설계)
echo "[7/9] ProteinMPNN..."
MPNN_DIR="$HOME/tools/ProteinMPNN"
if [ ! -d "$MPNN_DIR" ]; then
    mkdir -p "$HOME/tools"
    git clone https://github.com/dauparas/ProteinMPNN.git "$MPNN_DIR" --depth 1
    echo "  → $MPNN_DIR 에 설치됨"
else
    echo "  → 이미 존재: $MPNN_DIR"
fi

# 8. ColabFold 모델 가중치 다운로드 (선택)
echo "[8/9] ColabFold AlphaFold2 모델 다운로드 (약 5 GB, 건너뛰려면 Ctrl+C)..."
python3 -m colabfold.download 2>/dev/null || \
    python3 -c "
try:
    from colabfold.download import download_alphafold_params
    download_alphafold_params('multimer', '.')
    print('  → 다운로드 완료')
except Exception as e:
    print(f'  → 건너뜀: {e}')
"

# 9. 면역원성 분석 (T 세포 에피토프)
echo "[8/9] T 세포 에피토프 분석 (mhcflurry, epytope)..."
pip install mhcflurry epytope
# MHC-I 예측 모델 다운로드 (~2 GB, 시간 소요)
echo "  MHC-I 모델 다운로드 중 (~2 GB)..."
mhcflurry-downloads fetch || echo "  → 실패 시 나중에 'mhcflurry-downloads fetch' 재실행"

# 10. 유틸리티
echo "[9/9] 유틸리티 (numpy, scipy, pandas, matplotlib, jupyter)..."
pip install "numpy<2.0" "scipy>=1.10" pandas matplotlib seaborn jupyter

echo ""
echo "==========================================="
echo "         설치 완료 — 환경 검증"
echo "==========================================="
python3 - <<'VERIFY'
import sys
results = []

checks = [
    ("OpenMM",       "openmm"),
    ("ff19SB/OPC",   "openmmforcefields"),
    ("PDBFixer",     "pdbfixer"),
    ("MDTraj",    "mdtraj"),
    ("ESM-2",     "esm"),
    ("AbLang2",   "ablang2"),
    ("IgFold",    "igfold"),
    ("AntiFold",  "antifold"),
    ("ColabFold", "colabfold"),
    ("Boltz-2",   "boltz"),
    ("ANARCI",    "anarci"),
    ("Biopython", "Bio"),
]

for name, mod in checks:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "ok")
        results.append(f"  ✓ {name}: {ver}")
    except Exception as e:
        results.append(f"  ✗ {name}: {e}")

import torch
results.append(f"  ✓ PyTorch: {torch.__version__}")
results.append(f"  ✓ MPS(Metal): {torch.backends.mps.is_available()}")

import openmm
from openmm import Platform
platforms = [Platform.getPlatform(i).getName()
             for i in range(Platform.getNumPlatforms())]
results.append(f"  ✓ OpenMM platforms: {', '.join(platforms)}")

import os
results.append(
    f"  {'✓' if os.path.exists(os.path.expanduser('~/tools/ProteinMPNN')) else '✗'}"
    f" ProteinMPNN: ~/tools/ProteinMPNN"
)

print("\n".join(results))
VERIFY

echo ""
echo "==========================================="
echo "  ⚠️  CUDA 전용 도구 (M5 Mac 불가)"
echo "==========================================="
echo "  ✗ Chai-1r    : NVIDIA CUDA 필요 → RTX PC에서 실행"
echo "  ✗ RFdiffusion: NVIDIA CUDA 필요 → RTX PC에서 실행"
echo ""
echo "==========================================="
echo "  사용법"
echo "==========================================="
echo "  conda activate antibody"
echo "  cd ~/Antibody"
echo ""
echo "  # 서열 스코어링 (MPS 자동)"
echo "  python src/run_esm2_score.py --vh EVQLVES... --vl DIQMTQS..."
echo ""
echo "  # 항체 구조 예측"
echo "  python -c \"from igfold import IgFoldRunner; ...\""
echo ""
echo "  # 인간화"
echo "  python src/run_humanize.py --vh VH.fasta --vl VL.fasta"
echo ""
echo "  # MD 시뮬레이션 100ns (Metal GPU 자동 ~268 ns/day)"
echo "  python src/run_long_md.py --input complex.pdb --ns 100"
echo ""
echo "  # ColabFold 스크리닝 (CPU)"
echo "  python src/run_colabfold_screen.py --fasta candidates.fasta"
