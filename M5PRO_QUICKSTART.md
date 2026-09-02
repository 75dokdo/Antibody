# M5 Pro 실행 가이드

## 0. 먼저 pull (필수)
```bash
cd ~/Antibody
git pull origin claude/antibody-design-mldl7s
```

## 1. IgFold 구조 예측
```bash
# conda 없으면: brew install miniforge  or  pip3 install igfold
pip3 install torch igfold
python3 src/run_igfold.py
# 결과: fap_design/structures/<id>.pdb + igfold_summary.json
```

## 2. MHCflurry T세포 에피토프 (MHC-I)
```bash
pip3 install mhcflurry
python3 -c "from mhcflurry import Class1PresentationPredictor; Class1PresentationPredictor.load(download=True)"
python3 src/run_tcell_epitope.py
# 결과: fap_design/tcell/tcell_epitope_results.json (MHC-I 추가)
```

## 3. ColabFold 복합체 (LocalColabFold 필요)
```bash
# LocalColabFold 미설치 시: Google Colab 사용 (아래 대안)
bash fap_design/colabfold/run_colabfold.sh
python3 src/analyze_colabfold.py
```

### ColabFold 대안 — Google Colab
1. https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb
2. `fap_design/colabfold/top5_FAP_complex_batch.fasta` 업로드
3. model_type: alphafold2_multimer_v3, num_recycles: 3
4. 결과 다운로드 → `fap_design/colabfold/results/` 에 저장
5. `python3 src/analyze_colabfold.py`

## 4. MD (ColabFold 결과 후)
```bash
pip3 install openmm openmmforcefields pdbfixer
python3 src/run_md_openmm.py --batch --platform Metal --ns 100
```

## 5. 커밋 & 푸시
```bash
git add fap_design/
git commit -m "data: IgFold + ColabFold + MD 결과"
git push
```

---

## 오류 해결

| 오류 | 원인 | 해결 |
|------|------|------|
| `No such file or directory: run_colabfold.sh` | git pull 안 함 | `git pull origin claude/antibody-design-mldl7s` |
| `zsh: unknown file attribute: H` | 커밋 메시지 특수문자 | 아래 명령어 사용 |
| `zsh: command not found: conda` | conda 미설치 | `pip3` 직접 사용 |

### 커밋 메시지 특수문자 zsh 오류 해결:
```bash
git add fap_design/tcell/ && git commit -m 'data: T cell epitope results' && git push
# 또는
git add fap_design/ && git commit -m 'data: results' && git push origin claude/antibody-design-mldl7s
```
