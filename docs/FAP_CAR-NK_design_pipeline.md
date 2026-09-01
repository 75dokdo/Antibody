# FAP CAR-NK scFv 설계 파이프라인
## 특허 회피 + de novo 설계 전략

---

## 1. 표적 분석

### FAP (Fibroblast Activation Protein α)
- **UniProt**: Q12884 | **PDB**: 1Z68 (호모다이머, 2.5Å)
- **발현**: 암연관섬유아세포(CAF), 활성화 섬유아세포
- **정상조직**: 거의 없음 → 높은 치료 안전창
- **도메인 구조**:

```
[1-22] 신호펩타이드 → [23-50] 막관통 → [51-390] β-프로펠러 → [391-760] 촉매도메인
                                              ↑                        ↑
                                     FAP 특이적 (DPP4 비공유)    Ser624/Asp702/His734
```

---

## 2. 특허 회피 전략

### 2-1. 회피 대상 주요 특허

| 항체 | 특허 | 권리자 | 에피토프 영역 |
|------|------|--------|--------------|
| **Sibrotuzumab (F19)** | US6455494 | Boehringer Ingelheim | 촉매도메인 인근 |
| **FAP5 scFv** | US9017953 | Roche/Novartis | 촉매도메인 cleft |
| **28H1** | US10072079 | Bayer | ECD 전반 |
| **4B9** | EP2970422 | various | DPP cleft |
| **ESBATech E3** | US8716449 | Alcon | β-propeller |

### 2-2. 회피 방법

**① 에피토프 차별화** (가장 강력)
- 기존 항체: 주로 촉매도메인(391-760) 또는 DPP cleft 표적
- **본 설계**: β-프로펠러 Blade 6-7 (잔기 308-361) — 신규 에피토프
- FAP/DPP4 비공유 영역 → FAP 선택성 우수

**② CDR 서열 독창성**
- 기존 anti-FAP CDR과 BLAST 검색 → 유사도 <60% 확보
- de novo 설계(RFdiffusion) → 기존 CDR과 무관한 신규 서열 생성
- CDR-H3 길이 선택: 14-16 aa (기존 특허와 다른 길이)

**③ 프레임워크 + CDR 조합의 신규성**
- 트라스투주맙 FR + 신규 FAP CDR = 이중 신규성
- 트라스투주맙 FR 자체는 이미 공개되어 IP 부담 없음

**④ CAR 구성의 신규성** (NK 특화)
- 기존 특허: 대부분 CAR-T (CD3ζ + CD28 또는 4-1BB)
- **본 설계**: NK 특화 신호 도메인 (2B4 + CD3ζ 또는 NKp46-DAP12)

---

## 3. CAR-NK 구조 설계

### 3-1. CAR 도메인 구성

```
5'─[신호펩타이드]─[scFv]─[힌지]─[TM]─[공자극]─[CD3ζ]─3'
         ↓           ↓       ↓      ↓      ↓
     CD8α SS    VH-G4S3-VL  CD8α  NKp46  2B4
```

### 3-2. 도메인 서열

```
신호펩타이드: CD8α SS  (MALPVTALLLPLALLLHAARP)
힌지:         NKp46 extracellular stalk (NK 특화 - CAR-T 특허 회피)
              또는 CD8α hinge
막관통:       NKp46 TM  (NK 활성화 수용체 기반 - 특허 차별화)
공자극:       2B4 (CD244) - NK 특화 (CAR-T의 4-1BB/CD28과 차별화)
1차 신호:     CD3ζ ITAM × 3
```

### 3-3. 최종 CAR 구조 (권장)

```
[CD8α-SP]-[anti-FAP scFv (VH-G4S3-VL)]-[NKp46힌지]-[NKp46-TM]-[2B4]-[CD3ζ]
```

**특허 차별화 포인트**:
- NKp46 힌지/TM: CAR-T 특허에 없는 NK 특이 도메인
- 2B4 공자극: NK 세포 활성화 특화 (IL-15 시너지)
- scFv 신규 에피토프: β-프로펠러 Blade 6-7

---

## 4. scFv 설계 파이프라인

### Stage 1: FAP 구조 준비 (M5 Pro)

```bash
# PDB 1Z68 다운로드 + ECD 추출
python src/run_fap_scfv_design.py --download_pdb --out_dir fap_design

# 결과: fap_design/fap_ecd_A.pdb (잔기 51-760)
```

### Stage 2: 백본 생성 (RTX 3060 Ti — CUDA 필수)

```bash
# RFdiffusion: FAP β-프로펠러 표면에 결합하는 VH 백본 20개 생성
python src/run_fap_scfv_design.py --gen_rfdiffusion --out_dir fap_design

# 생성된 스크립트 RTX PC에서 실행:
bash fap_design/rfdiffusion/run_rfdiffusion.sh

# 핫스팟: A308, A310, A312, A314, A355, A357, A359, A361
# 출력: fap_design/rfdiffusion/fap_binder_0.pdb ~ fap_binder_19.pdb
```

### Stage 3: 서열 설계 (M5 Pro 또는 RTX)

```bash
# ProteinMPNN: 각 백본에서 3개 서열 (총 60개 후보)
bash fap_design/mpnn/run_proteinmpnn.sh

# 트라스투주맙 FR 이식
for candidate in fap_design/mpnn_seqs/*/; do
    python src/run_fap_scfv_design.py \
        --graft_cdrs H1 H2 H3 L1 L2 L3 \
        --out_dir fap_design
done
```

### Stage 4: 특허 회피 검증 (M5 Pro)

```bash
# 설계된 CDR을 기존 anti-FAP 항체 CDR과 BLAST 비교
python src/patent_check_blast.py \
    --query fap_design/candidates.fasta \
    --known_abs data/known_anti_fap_cdrs.fasta \
    --max_identity 60  # 60% 이하만 통과
```

### Stage 5: 구조 예측 및 복합체 검증 (M5 Pro)

```bash
# IgFold: scFv 단독 구조 예측 (~1분/후보)
python -c "
from igfold import IgFoldRunner
runner = IgFoldRunner()
runner.fold('fap_scfv_candidate.fasta', output_dir='fap_design/igfold/')
"

# ColabFold: FAP-scFv 복합체 구조 예측
python src/run_colabfold_screen.py \
    --fasta fap_design/candidates.fasta \
    --antigen_seq FAP_ECD_SEQ \
    --out_dir fap_design/colabfold

# Boltz-2: 추가 복합체 검증
boltz predict fap_design/candidates.fasta \
    --use_msa_server \
    --output_dir fap_design/boltz
```

### Stage 6: 서열 안정성 스코어 (M5 Pro)

```bash
# ESM-2 PLL 스코어링 + CDR 변이 스캔
python src/run_esm2_score.py \
    --fasta fap_design/candidates.fasta \
    --cdr_scan \
    --out_dir fap_design/esm2
```

### Stage 7: MD 검증 (M5 Pro — Metal GPU)

```bash
# 상위 5개 후보 복합체 100 ns MD
python src/run_long_md.py \
    --input fap_design/top_candidate_complex.pdb \
    --ns 100 \
    --ff ff19sb \
    --out_dir fap_design/md
```

---

## 5. 선정 기준 (필터링)

| 단계 | 지표 | 기준 |
|------|------|------|
| ① 특허 회피 | CDR BLAST 최대 유사도 | < 60% |
| ② 에피토프 | 핫스팟 접촉 잔기 | ≥ 5개 |
| ③ 복합체 구조 | ipTM (ColabFold) | ≥ 0.5 |
| ③ 복합체 구조 | pDockQ | ≥ 0.23 |
| ④ 서열 안정성 | ESM-2 PLL | 상위 30% |
| ⑤ 인간화 | 게르민라인 유사도 | ≥ 85% |
| ⑥ MD | RMSF CDR | < 3 Å |
| ⑥ MD | 에피토프 접촉 지속시간 | > 70% |

---

## 6. 일정 및 인프라 분담

```
RTX 3060 Ti (CUDA)          M5 Pro (Metal/MPS)
──────────────────────      ──────────────────────
Stage 2: RFdiffusion        Stage 1: FAP PDB 준비
Stage 3: ProteinMPNN        Stage 3: FR 이식
(병렬 가능)                  Stage 4: 특허 검증
                             Stage 5: IgFold/ColabFold
                             Stage 6: ESM-2 스코어
                             Stage 7: MD 100ns
                                      (~8.9h/후보)
```

---

## 7. 기대 결과물

1. **항-FAP scFv 서열** (VH/VL, Chothia CDR 주석 포함)
2. **FAP-scFv 복합체 구조** (PDB)
3. **CAR-NK 벡터 설계도** (도메인 서열 전체)
4. **특허 FTO 분석서** (기존 특허와 차별점)
5. **MD 안정성 데이터** (100 ns, ff19SB+OPC)
