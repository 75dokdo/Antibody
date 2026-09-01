# FAP CAR-NK scFv 설계 파이프라인
## M5 Pro 완결 · 특허 회피 · 인간화 · T 세포 에피토프 검증

---

## 1. 표적 분석

### FAP (Fibroblast Activation Protein α)
| 항목 | 내용 |
|------|------|
| UniProt | Q12884 |
| PDB | **1Z68** (호모다이머 2.5Å), 5XBT, 6B2H |
| 발현 | 암연관섬유아세포(CAF), 활성화 섬유아세포 |
| 정상조직 | 거의 없음 → 높은 안전창 |

**도메인 구조**
```
[1-22] SP → [23-50] TM → [51-390] β-프로펠러 → [391-760] 촉매도메인
                                  ↑                        ↑
                         FAP 특이적 (DPP4 비공유)     Ser624·Asp702·His734
```

---

## 2. 특허 회피 전략

### 회피 대상 주요 특허

| 항체 | 특허번호 | 에피토프 | 회피 방법 |
|------|----------|----------|-----------|
| Sibrotuzumab (F19) | US6455494 | 촉매도메인 인근 | 다른 에피토프 |
| FAP5 scFv | US9017953 | DPP cleft | 다른 에피토프 |
| 28H1 | US10072079 | ECD 전반 | CDR 서열 차별화 |
| 4B9 | EP2970422 | DPP cleft | 다른 에피토프 |

### 회피 전략 (3중 차별화)

| 전략 | 내용 |
|------|------|
| **① 신규 에피토프** | β-프로펠러 **Blade 6-7** (잔기 308-361) — 기존 특허와 다른 위치 |
| **② CDR 독창성** | AbLang2 + AntiFold de novo → 기존 항체와 유사도 < 60% |
| **③ CAR 도메인** | NKp46 힌지/TM + 2B4 공자극 — CAR-T 특허와 구조적 차별 |

---

## 3. CAR-NK 구조

```
5'─[SP]─[scFv]─[힌지]─[TM]─[공자극]─[CD3ζ]─3'
        │        │      │       │
    VH-G4S3-VL  NKp46  NKp46  2B4(CD244)
    (항-FAP)   stalk   TM     ← NK 특화 신호
```

**신호 도메인 서열 (GenBank 기반)**
```
신호펩타이드: CD8α SS   MALPVTALLLPLALLLHAARP
힌지:         NKp46    (NM_004829 기반 extracellular stalk, ~30 aa)
막관통:       NKp46 TM (특허 차별 — CAR-T의 CD8α TM과 다름)
공자극:       2B4      IYTYYGKQNFHMKPRAGGTKREKQALTERIKDNHIQNLPKTLPHPYFQ...
1차 신호:     CD3ζ     NQLYQPLKDREDDQYSSLGNQLRRQNQSKELISFLKQEKIPKETSQEK...
```

---

## 4. 전체 파이프라인 (M5 Pro 완결)

```
Stage 1  FAP 구조 준비          M5 Pro  PDB 1Z68 다운로드 · ECD 추출
Stage 2  CDR 초기 서열 생성     M5 Pro  AbLang2 + AntiFold (MPS)
Stage 3  트라스투주맙 FR 이식   M5 Pro  run_fap_scfv_design.py
Stage 4  구조 예측              M5 Pro  IgFold + ColabFold / Boltz-2
Stage 5  특허 회피 검증         M5 Pro  CDR BLAST 유사도 < 60%
Stage 6  인간화                 M5 Pro  run_humanize.py (ANARCI + ESM-2)
Stage 7  T 세포 에피토프        M5 Pro  mhcflurry (MHC-I) + IEDB API (MHC-II)
Stage 8  탈면역화               M5 Pro  FR 위치 보존적 치환 적용
Stage 9  MD 검증                M5 Pro  OpenMM ff19SB+OPC 100 ns (Metal GPU)
```

---

## 5. 단계별 실행 명령

### Stage 1 — FAP 구조 준비

```bash
python src/run_fap_scfv_design.py --download_pdb --out_dir fap_design
# → fap_design/fap_ecd_A.pdb (잔기 51-760, β-프로펠러 포함)
```

### Stage 2 — M5 Pro de novo CDR 생성

```python
# AbLang2: 항체 언어모델로 CDR 후보 생성 (MPS)
import ablang2, torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = ablang2.pretrained("heavy", device=device)

# 마스킹된 CDR 위치에서 후보 서열 샘플링
vh_template = "EVQLVESGGGLVQPGGSLRLSCAAS[MASK×7]YIHWVRQAPGKGLEWVARI" \
              "[MASK×5]YTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVY[MASK×12]" \
              "GFYAMDYWGQGTLVTVSS"
# AbLang2가 [MASK] 위치 채움 → FAP 결합 가능성 있는 CDR 후보 생성
```

```bash
# AntiFold: FAP 복합체 구조에서 최적 서열 역설계 (MPS)
antifold \
  --pdb_file fap_design/fap_ecd_A.pdb \
  --heavy_chain A \
  --num_seq_per_target 20 \
  --sampling_temp 0.2 \
  --out_dir fap_design/antifold_cdrs
```

### Stage 3 — 트라스투주맙 FR 이식

```bash
# 설계된 CDR을 트라스투주맙 FR에 이식
python src/run_fap_scfv_design.py \
  --graft_cdrs GFNIKDT YPTNG SRWGGDGF RASQDVNTAVA SASFLYSGVPS QQHYTTPPT \
  --out_dir fap_design

# → fap_design/fap_scfv_candidate.fasta
```

### Stage 4 — 구조 예측 및 복합체 검증

```bash
# IgFold: scFv 구조 예측 (~1분, MPS)
python -c "
from igfold import IgFoldRunner
runner = IgFoldRunner()
runner.fold(
    sequences={'H': VH_SEQ, 'L': VL_SEQ},
    output_pdb='fap_design/scfv_igfold.pdb'
)
"

# ColabFold: FAP + scFv 복합체 예측 (CPU, ~30-60분)
python src/run_colabfold_screen.py \
  --fasta fap_design/candidates.fasta \
  --out_dir fap_design/colabfold

# Boltz-2: 복합체 추가 검증 (MPS)
boltz predict fap_design/complex_input.yaml \
  --output_dir fap_design/boltz
```

### Stage 5 — 특허 회피 검증

```bash
# 설계 CDR vs 알려진 항-FAP 항체 CDR BLAST 비교
python -c "
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML

# CDR-H3 (가장 중요한 특허 차별 지점) 온라인 BLAST
record = SeqIO.read('fap_design/vh_cdr3.fasta', 'fasta')
result_handle = NCBIWWW.qblast('blastp', 'nr', record.seq)
blast_records = NCBIXML.read(result_handle)

for alignment in blast_records.alignments[:5]:
    hsp = alignment.hsps[0]
    identity_pct = hsp.identities / hsp.align_length * 100
    print(f'{alignment.title[:60]}: {identity_pct:.0f}% identity')
    if identity_pct > 60:
        print('  ⚠️  특허 위험 — CDR 재설계 필요')
"
```

### Stage 6 — 인간화

```bash
# run_humanize.py: ANARCI 넘버링 → 게르민라인 FR 검색 → CDR 이식 → 버니어존 백뮤테이션
python src/run_humanize.py \
  --vh VH_SEQUENCE \
  --vl VL_SEQUENCE \
  --out_dir fap_design/humanized

# 출력:
# humanized_variants.fasta   — 인간화 변이체 (최대 16종)
# scores.csv                 — ESM-2 PLL + 게르민라인 유사도
# grafting_report.json       — 이식 보고서
```

**인간화 선정 기준**
| 지표 | 기준 |
|------|------|
| 게르민라인 유사도 | VH ≥ 85%, VL ≥ 80% |
| ESM-2 PLL | 원래 대비 > -2.0 |
| CDR RMSD (vs 원래) | < 1.5 Å |

### Stage 7 — T 세포 에피토프 예측

```bash
# MHC-I (mhcflurry) + MHC-II (IEDB API)
python src/run_tcell_epitope.py \
  --vh HUMANIZED_VH \
  --vl HUMANIZED_VL \
  --out fap_design/tcell_epitopes.json
```

**분석 범위**

| 항목 | 설정 |
|------|------|
| MHC-I 대립유전자 | HLA-A\*01,02,03,24 / B\*07,15,35 / C\*07 (8종) |
| MHC-II 대립유전자 | DRB1\*01,03,04,07,11,13,15 (7종) |
| MHC-I 펩타이드 | 9-mer |
| MHC-II 펩타이드 | 15-mer |
| 위험 기준 | IC50 < 500 nM (MHC-I), < 1000 nM (MHC-II) |

### Stage 8 — 탈면역화 (필요 시)

```bash
# 에피토프가 FR 위치에 있으면 보존적 치환으로 제거
# CDR 위치는 건너뜀 (기능 유지)
# run_tcell_epitope.py 출력의 deimmunization_suggestions 참고

python src/run_humanize.py \
  --vh DEIMM_VH \
  --vl DEIMM_VL \
  --out_dir fap_design/deimmunized

# 탈면역화 후 재검증 (Stage 4-7 반복)
```

### Stage 9 — MD 검증

```bash
# OpenMM ff19SB + OPC, Metal GPU, HMR 4fs
python src/run_long_md.py \
  --input fap_design/top_complex.pdb \
  --ns 100 \
  --ff ff19sb \
  --out_dir fap_design/md

# 분석: RMSD, RMSF, 에피토프 접촉 지속시간
# 목표: CDR RMSF < 3Å, 핫스팟 접촉 > 70%
```

---

## 6. 선정 기준 (전체)

| 단계 | 지표 | 기준 |
|------|------|------|
| 특허 회피 | CDR BLAST 최대 유사도 | < 60% |
| 복합체 | ipTM (ColabFold) | ≥ 0.5 |
| 복합체 | pDockQ | ≥ 0.23 |
| 서열 안정성 | ESM-2 PLL | 상위 30% |
| 인간화 | 게르민라인 유사도 VH/VL | ≥ 85% / 80% |
| 면역원성 | MHC-I 강한 결합 수 | ≤ 1개 |
| 면역원성 | MHC-II 강한 결합 수 | ≤ 2개 |
| MD 안정성 | CDR RMSF | < 3 Å |
| MD 결합 | 핫스팟 접촉 지속 | > 70% |

---

## 7. 도구 및 설치 요약

```bash
# 모두 M5 Pro에서 설치 가능
conda activate antibody

# 구조 분석
conda install -c conda-forge openmm openmmforcefields pdbfixer mdtraj -y

# ML/AI
pip install torch  # MPS 자동
pip install fair-esm ablang2 igfold antifold

# 면역원성
pip install mhcflurry epytope
mhcflurry-downloads fetch      # MHC-I 모델 (~2 GB)

# 항체 분석
pip install anarci abnumber biopython==1.84

# 구조 예측
pip install "colabfold[cpu]" boltz
```

---

## 8. 예상 소요 시간 (M5 Pro 기준)

| 작업 | 시간 |
|------|------|
| Stage 1-3 (준비 + CDR 생성) | ~2-4시간 |
| Stage 4 IgFold 20후보 | ~20분 |
| Stage 4 ColabFold 복합체 20개 | ~10-20시간 |
| Stage 6 인간화 | ~30분 |
| Stage 7 에피토프 분석 | ~1시간 |
| Stage 9 MD 100 ns | ~9시간/후보 |
| **전체 (상위 5후보 MD)** | **~2-3일** |
