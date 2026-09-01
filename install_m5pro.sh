#!/bin/bash
# M5 Pro (Apple Silicon arm64) 항체 설계 도구 설치 스크립트
# macOS 15+ (Sequoia) 권장

set -e
echo "=== M5 Pro 항체 설계 환경 설치 ==="

# 1. Miniforge (arm64 Conda) 설치 확인
if ! command -v conda &>/dev/null; then
    echo "[1/8] Miniforge 설치 중..."
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
    conda init zsh
    echo ">>> 터미널 재시작 후 이 스크립트를 다시 실행하세요"
    exit 0
fi

# 2. Conda 환경 생성
echo "[2/8] Conda 환경 'antibody' 생성 (Python 3.11)..."
conda create -n antibody python=3.11 -y 2>/dev/null || echo "환경이 이미 존재합니다"
conda activate antibody || source activate antibody

# 3. OpenMM (Metal GPU 지원) + 핵심 도구
echo "[3/8] OpenMM + 핵심 패키지 설치 (Metal GPU 지원)..."
conda install -c conda-forge openmm pdbfixer mdtraj mdanalysis -y

# 4. ML/DL 도구 (PyTorch MPS 지원)
echo "[4/8] PyTorch (MPS Metal 지원)..."
pip install torch torchvision torchaudio

# 5. 항체 분석 도구
echo "[5/8] 항체 분석 도구..."
pip install fair-esm biopython==1.84 anarci

# 6. 구조 예측
echo "[6/8] 구조 예측 도구..."
# ColabFold (CPU 모드 - AlphaFold2 기반)
pip install "colabfold[cpu]"
# 모델 다운로드
python3 -c "from colabfold.download import download_alphafold_params; download_alphafold_params('multimer', '.')" 2>/dev/null || true

# Boltz-2 (MPS 지원)
pip install boltz

# 7. ProteinMPNN
echo "[7/8] ProteinMPNN..."
if [ ! -d "$HOME/tools/ProteinMPNN" ]; then
    mkdir -p "$HOME/tools"
    git clone https://github.com/dauparas/ProteinMPNN.git "$HOME/tools/ProteinMPNN" --depth 1
fi

# 8. 유틸리티
echo "[8/8] 유틸리티..."
pip install scipy numpy pandas matplotlib seaborn jupyter

echo ""
echo "=== 설치 완료 ==="
echo ""
echo "GPU 확인:"
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  MPS(Metal): {torch.backends.mps.is_available()}')
import openmm; from openmm import Platform
print(f'  OpenMM platforms:')
for i in range(Platform.getNumPlatforms()):
    p = Platform.getPlatform(i)
    print(f'    - {p.getName()}')
"

echo ""
echo "=== ⚠️  CUDA 전용 도구 (M5 Mac 불가) ==="
echo "  - Chai-1r  : NVIDIA CUDA 필요 → RTX PC에서만 실행"
echo "  - RFdiffusion : NVIDIA CUDA 필요 → RTX PC에서만 실행"
echo ""
echo "=== 사용법 ==="
echo "  conda activate antibody"
echo "  cd ~/Antibody"
echo "  python src/run_colabfold_screen.py ..."
echo "  python src/run_esm2_score.py ..."
echo "  python src/openmm_validate.py ...  # Metal GPU 자동 사용"
echo "  python src/run_long_md.py ...      # Metal GPU 자동 사용"
