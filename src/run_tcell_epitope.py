#!/usr/bin/env python3
"""
T 세포 에피토프 예측 및 탈면역화 (Deimmunization)
항체 VH/VL 서열에서 MHC-I / MHC-II 결합 펩타이드 탐지

MHC-I  : mhcflurry (로컬, M5 Pro CPU/MPS 지원)
MHC-II : IEDB REST API (온라인, 별도 설치 불필요)

목적: 면역원성(Immunogenicity) 예측 → CDR-외 위치 돌연변이로 탈면역화
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests

# ─── 트라스투주맙 FR 기반 설계 항체의 실제 순차 CDR 위치 ─────────────────────
# VH_FR3 말단 "YC"(C96) 포함 → CDR-H3는 순차 97번부터 시작 (최장 13aa → 109)
VH_CDR = {"H1": (26, 32), "H2": (52, 56), "H3": (97, 110)}
VL_CDR = {"L1": (24, 34), "L2": (50, 56), "L3": (89, 97)}

# ─── 분석할 HLA 대립유전자 ────────────────────────────────────────────────────
# MHC-I: 세계 인구의 >95% 커버하는 슈퍼타입 대표 8종
MHC_I_ALLELES = [
    "HLA-A*01:01", "HLA-A*02:01", "HLA-A*03:01", "HLA-A*24:02",
    "HLA-B*07:02", "HLA-B*15:01", "HLA-B*35:01", "HLA-C*07:02",
]
# MHC-II: EuroAmerican DRB1 패널 (면역원성에 더 관련)
MHC_II_ALLELES = [
    "DRB1*01:01", "DRB1*03:01", "DRB1*04:01", "DRB1*07:01",
    "DRB1*11:01", "DRB1*13:01", "DRB1*15:01",
]

# ─── 에피토프 경보 기준 ───────────────────────────────────────────────────────
MHCI_STRONG_BINDER_NM  = 50    # IC50 < 50  nM  → 강한 결합
MHCI_WEAK_BINDER_NM    = 500   # IC50 < 500 nM  → 주의
MHCII_STRONG_BINDER_NM = 50
MHCII_WEAK_BINDER_NM   = 1000  # MHC-II 기준 완화


# ─── CDR 마스크 ───────────────────────────────────────────────────────────────
def get_cdr_mask(seq_len: int, cdr_dict: dict) -> list:
    """CDR 위치 표시 (True=CDR, False=FR) — 1-indexed Chothia"""
    mask = [False] * seq_len
    for start, end in cdr_dict.values():
        for i in range(start - 1, min(end, seq_len)):
            mask[i] = True
    return mask


# ─── 슬라이딩 윈도우 펩타이드 생성 ──────────────────────────────────────────
def sliding_peptides(seq: str, length: int) -> list:
    """길이 length의 모든 서브서열 반환: (시작위치, 펩타이드)"""
    return [(i, seq[i: i + length]) for i in range(len(seq) - length + 1)]


# ─── MHC-I 예측 (mhcflurry) ─────────────────────────────────────────────────
def predict_mhci(seq: str, alleles: list = MHC_I_ALLELES,
                 lengths: list = [9]) -> list:
    """
    mhcflurry로 MHC-I 결합 예측
    반환: [{'pos':i, 'peptide':p, 'allele':a, 'affinity':nM, 'class':'strong'|'weak'|'none'}]
    """
    try:
        from mhcflurry import Class1PresentationPredictor
    except ImportError:
        print("[경고] mhcflurry 미설치: pip install mhcflurry && mhcflurry-downloads fetch")
        return []

    try:
        predictor = Class1PresentationPredictor.load()
    except Exception as e:
        print(f"[경고] mhcflurry 모델 로드 실패: {e}")
        print("  → mhcflurry-downloads fetch 실행 후 재시도")
        return []

    results = []
    for length in lengths:
        peptides_pos = sliding_peptides(seq, length)
        if not peptides_pos:
            continue
        positions, peptides = zip(*peptides_pos)

        df = predictor.predict(
            peptides=list(peptides),
            alleles=alleles,
            include_affinity_percentile=True,
        )
        for _, row in df.iterrows():
            aff = row.get("presentation_score", None) or row.get("affinity", 9999)
            # mhcflurry 버전에 따라 컬럼명 다를 수 있음
            if hasattr(row, "affinity"):
                aff = row["affinity"]
            pos_idx = list(peptides).index(row["peptide"]) if "peptide" in row else 0

            if aff < MHCI_STRONG_BINDER_NM:
                cls = "strong"
            elif aff < MHCI_WEAK_BINDER_NM:
                cls = "weak"
            else:
                continue  # 결합 없음 → 제외

            results.append({
                "pos":      positions[pos_idx] + 1,  # 1-indexed
                "peptide":  row.get("peptide", ""),
                "length":   length,
                "allele":   row.get("allele", ""),
                "affinity": round(float(aff), 1),
                "class":    cls,
                "mhc":      "I",
            })
    return results


# ─── MHC-II 예측 (IEDB REST API) ─────────────────────────────────────────────
def predict_mhcii(seq: str, alleles: list = MHC_II_ALLELES,
                  length: int = 15) -> list:
    """
    IEDB REST API로 MHC-II 결합 예측 (인터넷 연결 필요)
    http://tools-cluster-interface.iedb.org/tools_api/mhcii/
    """
    results = []
    iedb_url = "http://tools-cluster-interface.iedb.org/tools_api/mhcii/"

    allele_str = ",".join(alleles)
    payload = {
        "method":        "recommended",
        "sequence_text": seq,
        "allele":        allele_str,
        "length":        str(length),
    }
    try:
        resp = requests.post(iedb_url, data=payload, timeout=60)
        if resp.status_code != 200:
            print(f"[경고] IEDB API 오류 {resp.status_code} — MHC-II 건너뜀")
            return results

        lines = resp.text.strip().split("\n")
        header = lines[0].split("\t")
        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) < len(header):
                continue
            row = dict(zip(header, cols))
            ic50 = float(row.get("ic50", 99999))
            if ic50 > MHCII_WEAK_BINDER_NM:
                continue
            cls = "strong" if ic50 < MHCII_STRONG_BINDER_NM else "weak"
            results.append({
                "pos":      int(row.get("start", 0)) + 1,
                "peptide":  row.get("peptide", ""),
                "length":   length,
                "allele":   row.get("allele", ""),
                "affinity": round(ic50, 1),
                "class":    cls,
                "mhc":      "II",
            })
    except requests.exceptions.ConnectionError:
        print("[경고] IEDB API 연결 실패 — 인터넷 연결 확인")
    except Exception as e:
        print(f"[경고] IEDB API 오류: {e}")
    return results


# ─── 탈면역화 제안 ────────────────────────────────────────────────────────────
# 보존적 치환 그룹 (기능 유지 + 에피토프 파괴)
CONSERVATIVE_SUBSTITUTIONS = {
    "A": ["G", "S", "V"],
    "R": ["K", "Q"],
    "N": ["D", "Q", "S"],
    "D": ["E", "N"],
    "C": ["S", "A"],
    "Q": ["N", "E", "K"],
    "E": ["D", "Q"],
    "G": ["A", "S"],
    "H": ["N", "Q", "Y"],
    "I": ["L", "V"],
    "L": ["I", "V", "M"],
    "K": ["R", "Q"],
    "M": ["L", "I"],
    "F": ["Y", "W", "L"],
    "P": ["A", "G"],
    "S": ["T", "A", "N"],
    "T": ["S", "A"],
    "W": ["Y", "F"],
    "Y": ["F", "H", "W"],
    "V": ["I", "L", "A"],
}


def suggest_deimmunization(epitopes: list, seq: str, cdr_mask: list) -> list:
    """
    에피토프 위치에서 FR 잔기 돌연변이 제안
    CDR 위치는 건너뜀 (기능 유지)
    """
    suggestions = []
    visited = set()

    for ep in epitopes:
        for offset in range(ep["length"]):
            pos = ep["pos"] - 1 + offset  # 0-indexed
            if pos >= len(seq):
                continue
            if pos in visited:
                continue
            visited.add(pos)

            if cdr_mask[pos]:
                continue  # CDR — 건너뜀

            wt_aa = seq[pos]
            alts = CONSERVATIVE_SUBSTITUTIONS.get(wt_aa, [])
            suggestions.append({
                "position":  pos + 1,   # 1-indexed
                "wt":        wt_aa,
                "alts":      alts,
                "in_epitope": ep["peptide"],
                "mhc":       ep["mhc"],
                "allele":    ep["allele"],
                "affinity":  ep["affinity"],
            })
    return suggestions


# ─── 전체 분석 ────────────────────────────────────────────────────────────────
def analyze_immunogenicity(vh: str, vl: str,
                           run_mhci: bool = True,
                           run_mhcii: bool = True) -> dict:
    """VH / VL 서열에 대한 면역원성 전체 분석"""
    vh_cdr_mask = get_cdr_mask(len(vh), VH_CDR)
    vl_cdr_mask = get_cdr_mask(len(vl), VL_CDR)

    all_epitopes = []

    # MHC-I
    if run_mhci:
        print("[MHC-I] mhcflurry 예측 중...")
        vh_mhci = predict_mhci(vh)
        for e in vh_mhci:
            e["chain"] = "VH"
        vl_mhci = predict_mhci(vl)
        for e in vl_mhci:
            e["chain"] = "VL"
        all_epitopes += vh_mhci + vl_mhci
        print(f"  VH: {len(vh_mhci)}개, VL: {len(vl_mhci)}개")

    # MHC-II
    if run_mhcii:
        print("[MHC-II] IEDB API 예측 중...")
        vh_mhcii = predict_mhcii(vh)
        for e in vh_mhcii:
            e["chain"] = "VH"
        vl_mhcii = predict_mhcii(vl)
        for e in vl_mhcii:
            e["chain"] = "VL"
        all_epitopes += vh_mhcii + vl_mhcii
        print(f"  VH: {len(vh_mhcii)}개, VL: {len(vl_mhcii)}개")

    # 탈면역화 제안
    vh_suggests = suggest_deimmunization(
        [e for e in all_epitopes if e["chain"] == "VH"], vh, vh_cdr_mask
    )
    vl_suggests = suggest_deimmunization(
        [e for e in all_epitopes if e["chain"] == "VL"], vl, vl_cdr_mask
    )

    # 면역원성 점수 (강한 결합 수 / 전체 펩타이드 수)
    n_strong = sum(1 for e in all_epitopes if e["class"] == "strong")
    n_weak   = sum(1 for e in all_epitopes if e["class"] == "weak")
    total_peptides = len(sliding_peptides(vh, 9)) + len(sliding_peptides(vl, 9))
    immunogenicity_score = (n_strong * 2 + n_weak) / max(total_peptides, 1)

    return {
        "vh": vh,
        "vl": vl,
        "epitopes": all_epitopes,
        "n_strong": n_strong,
        "n_weak":   n_weak,
        "immunogenicity_score": round(immunogenicity_score, 4),
        "risk_level": (
            "HIGH"   if n_strong >= 3 else
            "MEDIUM" if n_strong >= 1 or n_weak >= 5 else
            "LOW"
        ),
        "deimmunization_suggestions": {
            "VH": vh_suggests,
            "VL": vl_suggests,
        },
    }


# ─── 결과 출력 ────────────────────────────────────────────────────────────────
def print_report(result: dict, vh_cdr_mask: list, vl_cdr_mask: list):
    risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    risk = result["risk_level"]

    print()
    print("=" * 62)
    print("  T 세포 에피토프 분석 결과")
    print("=" * 62)
    print(f"  면역원성 위험: {risk_color.get(risk, '')} {risk}")
    print(f"  강한 결합(< 50 nM):  {result['n_strong']}개")
    print(f"  약한 결합(< 500 nM): {result['n_weak']}개")
    print(f"  면역원성 점수: {result['immunogenicity_score']:.4f}")
    print()

    if result["epitopes"]:
        print("[ 검출된 에피토프 (IC50 nM) ]")
        for ep in sorted(result["epitopes"], key=lambda x: x["affinity"])[:15]:
            chain = ep["chain"]
            in_cdr = ""
            if chain == "VH":
                pos0 = ep["pos"] - 1
                if any(vh_cdr_mask[pos0 + k] for k in range(ep["length"])
                       if pos0 + k < len(vh_cdr_mask)):
                    in_cdr = " ⚠️ CDR 포함"
            print(f"  {ep['class'].upper():6} MHC-{ep['mhc']} {ep['allele']:<18}"
                  f" pos{ep['pos']:3d} {ep['peptide']}  {ep['affinity']:>7.1f} nM"
                  f"  [{chain}]{in_cdr}")
    else:
        print("  ✓ 에피토프 없음 (기준 이하)")

    print()
    deimm = result["deimmunization_suggestions"]
    vh_s = deimm["VH"]
    vl_s = deimm["VL"]
    if vh_s or vl_s:
        print("[ 탈면역화 제안 (FR 위치만) ]")
        for s in (vh_s + vl_s)[:10]:
            chain = "VH" if s in vh_s else "VL"
            alts_str = "/".join(s["alts"][:3]) if s["alts"] else "—"
            print(f"  {chain} pos{s['position']:3d} {s['wt']} → {alts_str}"
                  f"  (MHC-{s['mhc']} {s['affinity']:.0f} nM)")
    else:
        print("  ✓ 탈면역화 돌연변이 불필요")
    print("=" * 62)


# ─── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="VH/VL T 세포 에피토프 예측 및 탈면역화 제안"
    )
    parser.add_argument("--vh",   required=True, help="VH 아미노산 서열")
    parser.add_argument("--vl",   required=True, help="VL 아미노산 서열")
    parser.add_argument("--no_mhci",  action="store_true", help="MHC-I 예측 건너뜀")
    parser.add_argument("--no_mhcii", action="store_true", help="MHC-II 예측 건너뜀")
    parser.add_argument("--out",  default=None, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    vh = args.vh.strip().upper()
    vl = args.vl.strip().upper()

    print(f"VH ({len(vh)} aa): {vh[:30]}...")
    print(f"VL ({len(vl)} aa): {vl[:30]}...")

    result = analyze_immunogenicity(
        vh, vl,
        run_mhci=not args.no_mhci,
        run_mhcii=not args.no_mhcii,
    )

    vh_cdr_mask = get_cdr_mask(len(vh), VH_CDR)
    vl_cdr_mask = get_cdr_mask(len(vl), VL_CDR)
    print_report(result, vh_cdr_mask, vl_cdr_mask)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[저장] {out_path}")


if __name__ == "__main__":
    main()
