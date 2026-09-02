#!/bin/bash
# ColabFold FAP ECD + scFv 복합체 구조 예측
# LocalColabFold v1.5+ 필요: https://github.com/YoshitakaMo/localcolabfold
# GPU 권장 (M5 Pro Metal 또는 CUDA)
#
# 설치: bash <(curl -fsSL https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh)
# 또는 conda install -c conda-forge colabfold

set -e

COLABFOLD_BIN=$(which colabfold_batch 2>/dev/null || echo "$HOME/localcolabfold/colabfold-conda/bin/colabfold_batch")

if [ ! -f "$COLABFOLD_BIN" ]; then
    echo "[ERROR] colabfold_batch 미설치. LocalColabFold 설치 후 재시도."
    echo "  설치: https://github.com/YoshitakaMo/localcolabfold"
    exit 1
fi

echo "[ColabFold] FAP ECD + scFv 복합체 예측 시작"
echo "  입력: fap_design/colabfold/top5_FAP_complex_batch.fasta"
echo "  출력: fap_design/colabfold/results/"
echo ""

mkdir -p fap_design/colabfold/results

# 배치 실행 (Top 5 한번에)
$COLABFOLD_BIN \
    fap_design/colabfold/top5_FAP_complex_batch.fasta \
    fap_design/colabfold/results \
    --num-recycle 3 \
    --model-type alphafold2_multimer_v3 \
    --use-gpu-relax \
    --num-models 5 \
    --rank ipTM

echo ""
echo "[완료] 결과: fap_design/colabfold/results/"
echo ""
echo "다음 단계: python3 src/analyze_colabfold.py"
