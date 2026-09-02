#!/bin/bash
# IgFold conda 환경 설정 (M5 Pro Apple Silicon)
conda create -n igfold python=3.10 -y
conda activate igfold
pip install torch torchvision torchaudio
pip install igfold
python src/run_igfold.py
