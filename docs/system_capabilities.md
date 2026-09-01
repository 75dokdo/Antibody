# 이 시스템에서 실제로 가능한 항체 설계 도구

작성일: 2026-09-01 · **모든 항목은 추측이 아니라 이 컨테이너에서 직접 설치·실행한 결과입니다.**

## 하드웨어 / 네트워크 제약

| 항목 | 상태 | 확인 방법 |
|---|---|---|
| GPU | **없음** | `nvidia-smi` 미존재 |
| CPU / RAM | 4 core / 15 GB | `nproc`, `free -g` |
| PyPI, apt(ubuntu) | 도달 가능 | 설치 성공 |
| github.com, raw.githubusercontent.com | **도달 가능** | clone 성공 |
| huggingface.co, zenodo.org, files.ipd.uw.edu | **차단** | curl 000 / ProxyError 403 |
| patents.google.com, uspto, ncbi, rcsb, nature, opig | **차단** | egress 정책 |

핵심 함의: **모델 가중치가 GitHub 저장소 안에 있으면 쓸 수 있고, HuggingFace·Zenodo·IPD에 있으면 못 씁니다.**

---

## ✅ 지금 바로 쓸 수 있는 것 (실행 검증 완료)

| 도구 | 버전 | 용도 | 검증 결과 |
|---|---|---|---|
| **ProteinMPNN** | v_48_020 | 주어진 백본에 서열 설계 | **2서열/1.24초 (CPU)** 실행 성공, 가중치 180 MB 확보 |
| **LightDock** | 3.0 | CPU 단백질-단백질 도킹, **잔기 restraint 지원** | 설치·CLI 확인 |
| **ANARCI** (microANARCI) | - | IMGT/Kabat/Chothia 번호매기기, germline 배정 | trastuzumab VH→human H 정확 판정 |
| **NCBI BLAST+** | 2.12.0 | 서열 유사도 (특허 검사에 사용 중) | 본 프로젝트에서 가동 중 |
| **sadie-antibody** | - | IMGT germline 서열 DB (인간/마우스/붉은털원숭이 등) | IGHV3-23*01 등 조회 성공 |
| **OpenMM + pdbfixer** | 8.6 | 에너지 최소화, CPU MD, 구조 보수 | import 성공 |
| **Biopython** | 1.88 | 구조 파싱, SASA(Shrake-Rupley) | 본 프로젝트에서 가동 중 |
| **freesasa / ProDy / pdb-tools** | 2.6.1 등 | 표면적, 구조 분석, PDB 조작 | import 성공 |
| **PyTorch** | 2.13.0 | CPU 추론 | import 성공 |
| 본 저장소 자체 도구 | - | 특허 에피토프 스캔, 표면 패치, scFv 입력 생성 | 전부 가동 |

### ⚠️ 주의: abnumber는 쓰지 마십시오
`abnumber`도 설치되지만, **이 빌드에서 카파 경사슬 CDR을 한 잔기씩 왼쪽으로 밀어 반환합니다**
(VL CDR2를 `SAS`가 아니라 `YSA`로 반환). 중사슬은 정상입니다.
ANARCI 원본 번호는 정확하므로, 본 저장소는 ANARCI를 직접 호출합니다.
설계 시 고정할 프레임워크 위치가 어긋나는 실질적 버그이므로 교차검증이 필요합니다.

---

## ❌ 이 시스템에서 불가능한 것

| 도구 | 불가 사유 | 확인 |
|---|---|---|
| **RFdiffusion / RFantibody** | 가중치가 `files.ipd.uw.edu` → 차단 + GPU 필요 | 호스트 도달 불가 |
| **ABodyBuilder2 / ImmuneBuilder** | 가중치가 `zenodo.org` → 차단 | 실행 시 ProxyError 403 |
| **IgFold, ESMFold, AbLang 등** | 가중치가 HuggingFace → 차단 | 호스트 도달 불가 |
| **AlphaFold2/3, AF-Multimer** | GPU 없음 + 가중치 배포처 차단 | - |
| **HDOCK / ClusPro / HADDOCK** | 웹 서비스, egress 차단 | - |
| **Rosetta / PyRosetta / FoldX** | 별도 라이선스 필요 | - |

---

## 결론: 두 갈래로 나눠 진행하는 것이 맞습니다

### 이 시스템에서 완결 가능한 단계
1. 에피토프 선정 + 특허 스크리닝 — **완료**
2. 표면 패치 분석 — **완료**
3. scFv 스캐폴드 조립(허셉틴 프레임워크) + 도킹 입력 생성 — **완료**
4. **LightDock으로 에피토프 restraint 도킹** — 가능, 미실행
5. **ProteinMPNN으로 계면 서열 설계** — 가능, 미실행
6. OpenMM 최소화 + 설계 결과 특허 재검사 — 가능

즉 **"기존 백본에 서열을 얹는" 경로는 이 시스템에서 끝까지 갈 수 있습니다.**

### GPU 장비가 필요한 단계
- CDR 백본 자체를 새로 생성 (RFdiffusion/RFantibody)
- 설계 항체 구조 예측 검증 (ABodyBuilder2 / IgFold)
- 복합체 구조 검증 (AlphaFold-Multimer / Boltz)

→ `design/run_design.sh`가 이 단계용으로 준비되어 있습니다. 가중치는 GPU 장비에서 받으십시오.
