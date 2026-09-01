# GPC3 항체 설계 최종 보고서

## 1. 도킹 (LightDock)

| 항목 | 값 |
|---|---|
| 도구 | LightDock 0.9.4 (DFIRE2 scoring) |
| 수용체 | `design/target_patch273.pdb` — Region B 패치, 53잔기 (성숙 256–387) |
| 리간드 | `design/trastuzumab_fv.pdb` — trastuzumab Fv (H+L chain, 227잔기) |
| Hotspot 제약 | A267,A270,A273,A274,A277,A372,A373,A375,A376,A379,A380 (11잔기) |
| 최적 스웜 | 90 (score -784.511, DFIRE2) |
| 인터페이스 확인 | Hotspot 11잔기 전원 < 5 Å 접촉 확인 |

최적 도킹 포즈: `design/docked_complex.pdb`

## 2. 서열 설계 (ProteinMPNN)

| 항목 | 값 |
|---|---|
| 도구 | ProteinMPNN v_48_020 |
| 프레임워크 | trastuzumab (hu4D5) VH-(G4S)₃-VL, 242 aa |
| 고정 위치 | VH 91개 + VL 89개 (framework 잔기) |
| 설계 위치 | VH CDR 29개 + VL CDR 18개 = 47개 |
| 온도 | 0.1 (보수적 설계) |
| 생성 수 | 10개 |

## 3. 상위 설계 순위 (ProteinMPNN score 기준, 낮을수록 우수)

| 순위 | Sample | Score | CDR-H1 | CDR-H2 | CDR-H3 | CDR-L1 | CDR-L2 | CDR-L3 |
|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 1.4907 | GFNIADTW | IYPADQTT | ATDLGSGSLGLAV | GLGGGG | GAL | AGGGLIPIG |
| 2 | 2 | 1.4931 | GFSIADTW | IYPADQST | ATDLGSGFLGLEV | GSGGTG | GLS | IGAGADPIG |
| 3 | 9 | 1.5152 | GFNIADTW | IHPADQTT | ATDLGPDFKGLEV | GAGGAG | GID | ALAGKEPIG |
| 4 | 6 | 1.5168 | GFSISDTW | INPADQTT | ATDLGPEAKGLAV | GAGGAG | GAD | LGGGKEPLG |
| 5 | 3 | 1.5187 | GFNIADTW | ILPSNQAT | ATDLGSGFLGLAV | GLGGEG | GIS | AGGGKVPIG |

## 4. 1순위 설계 (Sample 8)

**scFv 전서열 (242 aa):**
```
EVQLVESGGGLVQPGGSLRLSCAASGFNIADTWIHWVRQAPGKGLEWVARIYPADQTTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCATDLGSGSLGLAVWGQGTLVTVSGGGGSGGGGSGGGGSDIQMTQSPSSLSASVGDRVTITCRASGLGGGGVAWYQQKPGKAPKLLIYGALFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCAGGGLIPIGFGQGTKVEIKR
```

**VH (120 aa):**
```
EVQLVESGGGLVQPGGSLRLSCAASGFNIADTWIHWVRQAPGKGLEWVARIYPADQTTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCATDLGSGSLGLAVWGQGTLVTVS
```

**VL (107 aa):**
```
DIQMTQSPSSLSASVGDRVTITCRASGLGGGGVAWYQQKPGKAPKLLIYGALFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCAGGGLIPIGFGQGTKVEIKR
```

| CDR | 서열 | 길이 |
|---|---|---|
| CDR-H1 (IMGT 26-33) | `GFNIADTW` | 8 aa |
| CDR-H2 (IMGT 51-58) | `IYPADQTT` | 8 aa |
| CDR-H3 (IMGT 97-109) | `ATDLGSGSLGLAV` | 13 aa |
| CDR-L1 (IMGT 27-32) | `GLGGGG` | 6 aa |
| CDR-L2 (IMGT 50-52) | `GAL` | 3 aa |
| CDR-L3 (IMGT 89-97) | `AGGGLIPIG` | 9 aa |

## 5. 특허 유사도 검사

```
python3 src/patent_seq_similarity.py query --fasta design/mpnn_output/top5_cdrs.fasta --show-low
→ no similarity to any indexed claimed epitope
```

전체 scFv 서열 BLAST (--show-low 포함):
- LOW (trivial, 2잔기 dipeptide): HS20 epitope에 1건 — 의미 없는 우연 일치

**결론: 상위 5개 설계 모두 CLEAR 판정**

## 6. 다음 단계 (권고)

1. **AlphaFold2/ESMFold 구조 예측** — GPU 환경에서 최종 모델 예측
2. **Rosetta FastRelax** — 도킹 포즈 에너지 최소화
3. **IMGT/DomainGapAlign** — VH/VL germline 배정 및 humanization score
4. **실험 발현** — 상위 3-5개 E. coli/CHO 소규모 발현
5. **법적 FTO 의견** — CDR 서열에 대한 GC33·YP7·HN3 특허 청구항 비교 (변리사 필요)

## 7. 파일 목록

| 파일 | 설명 |
|---|---|
| `design/trastuzumab_fv.pdb` | 출발 Fv 구조 (H+L chain) |
| `design/docked_complex.pdb` | 최적 도킹 포즈 (A+H+L chain) |
| `design/mpnn_output/seqs/docked_complex.fa` | ProteinMPNN 전체 출력 |
| `results/mpnn_designs.json` | 10개 설계 상세 결과 (JSON) |
| `results/gpc3_designs.fasta` | 10개 scFv FASTA |
| `design/mpnn_output/top5_cdrs.fasta` | 상위 5개 CDR 서열 (특허 검사용) |

---
*본 도구는 연구용 트리아지 보조 수단이며 법적 FTO 의견이 아닙니다.*

---

## 8. 대규모 다온도 설계 (1,000개)

### 실행 설정

| 온도 | 서열 수 | 특성 |
|---|---|---|
| 0.05 | 200 | 극보수적 (score 1.43–1.56) |
| 0.10 | 200 | 보수적 (score 1.45–1.59) |
| 0.20 | 200 | 중간 (score 1.47–1.72) |
| 0.30 | 200 | 탐색적 (score 1.55–1.87) |
| 0.50 | 200 | 창의적 (score 1.70–2.17) |

### 필터링 결과

| 단계 | 개수 |
|---|---|
| 생성 | 1,000개 |
| 스코어 필터 (T≤0.3, 상위 300) | 300개 |
| CDR-H3+L3 중복 제거 | 267개 |
| 특허 MODERATE 제거 | 265개 |
| **최종 실험 후보 (96-well 1판)** | **96개** |

### 다양성

- CDR-H3 유니크 서열: **785종 / 1,000개**
- CDR-L3 유니크 서열: **765종 / 1,000개**

### 최종 96개 온도별 구성

- T=0.05: 72개 (보수적, 높은 신뢰도)
- T=0.10: 23개 (중간)
- T=0.20: 1개

### 상위 5개 후보

| 순위 | Score | CDR-H1 | CDR-H2 | CDR-H3 | CDR-L3 |
|---|---|---|---|---|---|
| 1 | 1.4311 | GFNIADTW | IKPADGTT | ATDLGPGFLGLEV | AGGGVSPIG |
| 2 | 1.4412 | GFNIADTW | IKPADGTT | ATDLGSGFDGLAV | AGGGAEPIG |
| 3 | 1.4453 | GFNIADTW | INPADGST | ATDLGPGFKGLAV | AGGGAEPIG |
| 4 | 1.4457 | GFNIADTW | INPADQTT | AIDYGSSFKGLAY | LGAGKEPIG |
| 5 | 1.4489 | GFNIADTW | IKPADGTT | ATDLGPGFLGLSV | AGGGAEPIG |

**파일**: `results/gpc3_top96.fasta`, `results/gpc3_top96.json`
