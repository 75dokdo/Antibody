# GPC3 항체 설계 — 특허 회피 에피토프 스크리닝

간세포암(HCC) 표적 GPC3 / Glypican-3 (UniProt P51654)에 대한 구조 기반 항체 설계 중,
**기존 특허 에피토프를 회피한 타겟 부위를 선정**하기 위한 파이프라인입니다.

## 핵심 결과

**Region B (성숙 190-300 = UniProt 214-324)** 를 1순위로 권고합니다.
잔기 수준 특허 충돌 0건, 평균 pLDDT 90.9, 기능적 hotspot 없음.

**Region D (성숙 460-520)** 는 YP7·GC33 등 5개 항체 에피토프와 충돌하며
pLDDT 38.8의 유연 영역이라 사용 불가입니다.

전체 분석: [`results/epitope_selection_report.md`](results/epitope_selection_report.md)

## ⚠️ 번호 체계 주의

구조 파일은 **성숙단백질 번호**, 논문·특허는 **UniProt 전장 번호**를 씁니다.

```
UniProt = 성숙(PDB) + 24
```

보정 없이 비교하면 24잔기가 어긋나 결론이 뒤집힙니다. 파이프라인에 내장되어 있습니다.

## 구성

```
data/
  structures/     GPC3 ectodomain 모델 + 후보 영역 A/B/C/D
  patents/        gpc3_patent_epitopes.json  — 큐레이션된 특허 에피토프 DB
                  blastdb/                   — BLAST 검색용 인덱스 (생성물)
src/
  gpc3_lib.py                번호 변환, 구조 로딩, SASA 계산
  epitope_fto_scan.py        좌표 기반 특허 충돌 스캔
  patent_seq_similarity.py   BLAST 기반 서열 유사도 검사
results/
  epitope_selection_report.md   최종 리포트
  fto_scan.json                 기계 판독용 전체 결과
```

## 설치

```bash
apt-get install -y ncbi-blast+     # BLAST+ 2.12.0
pip3 install biopython             # 1.88
```

## 사용법

### 1. 좌표 기반 특허 충돌 스캔

후보 영역이 알려진 특허 에피토프와 잔기 수준에서 겹치는지 검사합니다.
용매 노출도(SASA)는 **온전한 ectodomain에서** 계산합니다 — 영역을 잘라내어 계산하면
원래 매몰된 잔기가 노출된 것처럼 보이기 때문입니다.

```bash
python3 src/epitope_fto_scan.py                        # 전체 영역
python3 src/epitope_fto_scan.py --json results/fto_scan.json
python3 src/epitope_fto_scan.py --regions data/structures/GPC3_regionB.pdb
```

판정 등급: `CLEAR` → `LOW` → `MEDIUM` → `HIGH RISK`.
N-lobe 전체처럼 도메인 수준으로만 알려진 에피토프는 별도 `advisories`로 분리합니다
(수백 잔기를 한꺼번에 차단 처리하면 실제 신호가 묻히기 때문).

### 2. 서열 유사도 검사

좌표 스캔이 놓치는 두 경우를 잡습니다 — 항원의 다른 위치가 특허 에피토프를
서열로 흉내내는 경우, 그리고 번호 없이 서열만 있는 설계물을 검사하는 경우.

```bash
python3 src/patent_seq_similarity.py build              # DB 구축 (최초 1회)
python3 src/patent_seq_similarity.py query --seq PKDNEISTFH
python3 src/patent_seq_similarity.py query --fasta my_designs.fasta
python3 src/patent_seq_similarity.py selfscan --window 15 --step 5
```

판정 등급: `CRITICAL` (≥90% 동일, 8잔기 이상) → `HIGH` → `MODERATE` → `LOW`.
짧은 펩타이드 검색이므로 `blastp-short` + PAM30 설정을 사용합니다.

## 한계

- 특허 원문을 직접 읽지 못했습니다(네트워크 정책상 차단). 에피토프 좌표는 검색
  스니펫 기반 2차 정보이며, DB 각 항목에 `evidence` 등급을 표기했습니다.
- DB에는 **항원 에피토프만** 있고 항체 CDR 서열은 없습니다. 실제 침해 판단의 핵심은
  CDR 청구항이므로, GC33·YP7·HN3의 VH/VL 서열 확보가 다음 우선순위입니다.
- **본 도구는 연구용 트리아지 보조 수단이며 법적 FTO 의견이 아닙니다.**
  상업화 시 변리사 검토가 반드시 필요합니다.
