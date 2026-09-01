#!/usr/bin/env python3
"""
항체 scFv 개발가능성(Developability) 평가 도구
M5 Pro CPU/MPS 완결 — CUDA 불필요

평가 항목:
  1. 물리화학적 특성    MW, pI, GRAVY, 순전하
  2. 화학적 취약부위    탈아미드화, 이성화, 산화, N-글리코실화, 유리 Cys
  3. 집합 위험도        소수성 패치, 양전하 패치 (CDR)
  4. CDR 프로파일       CDR 길이, 희귀 서열, 이황화결합 위험
  5. 용해도 예측        CamSol-intrinsic 간소화 버전
  6. 인간성 점수        BioPhi OASis / Sapiens (설치 시)
  7. 종합 개발가능성    가중 점수 + 위험 등급
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

# ─── 아미노산 물성 테이블 ─────────────────────────────────────────────────────
AA_MW = {
    "A": 89.09,  "R": 174.20, "N": 132.12, "D": 133.10,
    "C": 121.16, "Q": 146.15, "E": 147.13, "G":  75.03,
    "H": 155.16, "I": 131.17, "L": 131.17, "K": 146.19,
    "M": 149.21, "F": 165.19, "P": 115.13, "S": 105.09,
    "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}
# pKa (Henderson-Hasselbalch용)
AA_PKA = {
    "D": 3.86, "E": 4.07, "H": 6.04, "C": 8.14,
    "Y": 10.46, "K": 10.54, "R": 12.48,
}
# Kyte-Doolittle 소수성 척도
KD_HYDRO = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5,
    "C":  2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
    "H": -3.2, "I":  4.5, "L":  3.8, "K": -3.9,
    "M":  1.9, "F":  2.8, "P": -1.6, "S": -0.8,
    "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

# 트라스투주맙 FR 기반 설계 항체의 실제 순차 CDR 위치 (1-indexed)
# VH_FR3 말단 "YC"(C96) 포함 → CDR-H3는 순차 97번부터 시작
VH_CDR = {"H1": (26, 32), "H2": (52, 56), "H3": (97, 110)}
VL_CDR = {"L1": (24, 34), "L2": (50, 56), "L3": (89, 97)}

# 화학적 취약 모티프
LIABILITY_MOTIFS = {
    "deamidation":    (r"N[GSTAH]",   "HIGH",   "탈아미드화 위험 (NG/NS/NT/NH)"),
    "isomerization":  (r"D[GSTPA]",   "HIGH",   "이성화 위험 (DG/DS/DP/DT/DA)"),
    "oxidation_M":    (r"M",          "MEDIUM", "메티오닌 산화 위험"),
    "oxidation_W":    (r"W",          "MEDIUM", "트립토판 산화 위험"),
    "n_glycosylation":(r"N[^P][ST]",  "MEDIUM", "N-연결 글리코실화 위험 (NxS/T)"),
    "dp_cleavage":    (r"DP",         "LOW",    "Asp-Pro 펩타이드 결합 가수분해"),
    "cd_cleavage":    (r"CD",         "LOW",    "Cys-Asp 결합 취약"),
}


# ─── CDR 마스크 ───────────────────────────────────────────────────────────────
def get_cdr_positions(seq_len: int, cdr_dict: dict) -> set:
    pos = set()
    for start, end in cdr_dict.values():
        pos |= set(range(start - 1, min(end, seq_len)))
    return pos


# ─── 1. 물리화학적 특성 ───────────────────────────────────────────────────────
def calc_biophysical(seq: str) -> dict:
    """MW, pI, GRAVY, 순전하, 불안정성 지수"""
    # 분자량
    mw = sum(AA_MW.get(aa, 111.1) for aa in seq) - 18.02 * (len(seq) - 1)

    # pI (이진 탐색)
    def net_charge(seq, pH):
        charge = 0.0
        # N-말단 (+)
        charge += 1 / (1 + 10 ** (pH - 8.0))
        # C-말단 (-)
        charge -= 1 / (1 + 10 ** (3.1 - pH))
        for aa in seq:
            if aa in ("K", "R", "H"):
                pka = AA_PKA.get(aa, 10.0)
                charge += 1 / (1 + 10 ** (pH - pka))
            elif aa in ("D", "E", "C", "Y"):
                pka = AA_PKA.get(aa, 4.0)
                charge -= 1 / (1 + 10 ** (pka - pH))
        return charge

    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    pi = round((lo + hi) / 2, 2)

    # 순전하 pH 7.4
    charge_74 = round(net_charge(seq, 7.4), 2)

    # GRAVY (Grand Average of Hydropathicity)
    gravy = round(sum(KD_HYDRO.get(aa, 0) for aa in seq) / len(seq), 3)

    # 불안정성 지수 (Guruprasad 1990) — 단순화 버전
    instability_pairs = {
        "WW": 1.0, "WC": 1.0, "WM": 24.7, "WH": 24.7, "WR": 1.0,
        "WY": 1.0, "WF": 1.0, "FY": 1.0, "YW": 1.0, "NY": 1.0,
        "DY": 1.0, "GY": -6.3,
    }
    insta = 0.0
    for i in range(len(seq) - 1):
        pair = seq[i] + seq[i + 1]
        insta += instability_pairs.get(pair, 0.0)
    instability_index = round(insta * 10 / len(seq), 2)

    return {
        "mw_kda":          round(mw / 1000, 2),
        "pi":              pi,
        "net_charge_ph74": charge_74,
        "gravy":           gravy,
        "instability_index": instability_index,
        "stable":          instability_index < 40,
    }


# ─── 2. 화학적 취약부위 ───────────────────────────────────────────────────────
def find_liabilities(seq: str, cdr_pos: set, chain: str = "VH") -> list:
    """서열에서 화학적 취약 모티프 탐지"""
    findings = []
    for key, (pattern, severity, desc) in LIABILITY_MOTIFS.items():
        for m in re.finditer(pattern, seq):
            start = m.start()
            in_cdr = any(p in cdr_pos for p in range(start, start + len(m.group())))
            findings.append({
                "type":     key,
                "motif":    m.group(),
                "position": start + 1,          # 1-indexed
                "in_cdr":   in_cdr,
                "severity": severity if in_cdr else "LOW",
                "desc":     desc,
                "chain":    chain,
                "risk":     "HIGH" if (in_cdr and severity == "HIGH") else
                            "MEDIUM" if (in_cdr and severity == "MEDIUM") else "LOW",
            })
    return findings


def check_cysteine(seq: str, chain: str) -> dict:
    """
    VH/VL 내 Cys 수 및 유리 Cys 위험 평가
    보존 이황화결합: 정수 번호가 아닌 위치(순서)로 판단
    - VH: 첫 번째 Cys (FR1 말단, ~22번) + 마지막에서 두 번째 Cys (CDR-H3 앞)
    - VL: 첫 번째 Cys (FR1 말단, ~23번) + 마지막 Cys (CDR-L3 앞)
    """
    cys_positions = [i + 1 for i, aa in enumerate(seq) if aa == "C"]
    total = len(cys_positions)

    # 보존 이황화결합: 첫 번째와 두 번째 Cys가 쌍 (2개 이상일 때)
    if total >= 2:
        # VH: 보통 2개 (pos ~22, ~96); VL: 보통 2개 (pos ~23, ~88)
        conserved_pair = {cys_positions[0], cys_positions[1]}
        unpaired = [p for p in cys_positions if p not in conserved_pair]
    elif total == 1:
        conserved_pair = set()
        unpaired = cys_positions  # 단독 Cys → 유리
    else:
        conserved_pair = set()
        unpaired = []

    return {
        "total_cys":            total,
        "positions":            cys_positions,
        "conserved_disulfide":  sorted(conserved_pair),
        "unpaired_cys":         unpaired,
        "risk":                 "HIGH" if unpaired else "OK",
    }


# ─── 3. 집합 위험도 (Aggregation Risk) ───────────────────────────────────────
HYDROPHOBIC_AAS = set("VILMFYW")
POSITIVE_AAS    = set("KRH")

def calc_aggregation_risk(seq: str, cdr_pos: set) -> dict:
    """CDR 영역 소수성/양전하 패치 분석"""
    window = 5
    hydro_patches, pos_patches = [], []

    for i in range(len(seq) - window + 1):
        peptide = seq[i: i + window]
        in_cdr  = any(p in cdr_pos for p in range(i, i + window))
        hydro   = sum(1 for aa in peptide if aa in HYDROPHOBIC_AAS) / window
        pos     = sum(1 for aa in peptide if aa in POSITIVE_AAS)    / window

        if hydro >= 0.8 and in_cdr:   # 80% 임계값 (5-mer 중 4개 이상 소수성)
            hydro_patches.append({
                "position": i + 1, "peptide": peptide,
                "hydro_fraction": round(hydro, 2), "in_cdr": True,
            })
        if pos >= 0.4 and in_cdr:
            pos_patches.append({
                "position": i + 1, "peptide": peptide,
                "pos_fraction": round(pos, 2), "in_cdr": True,
            })

    risk = "HIGH" if len(hydro_patches) >= 3 or len(pos_patches) >= 2 else \
           "MEDIUM" if hydro_patches or pos_patches else "LOW"

    return {
        "hydrophobic_patches": hydro_patches[:5],
        "positive_patches":    pos_patches[:5],
        "n_hydro_patches":     len(hydro_patches),
        "n_pos_patches":       len(pos_patches),
        "aggregation_risk":    risk,
    }


# ─── 4. CDR 프로파일 ──────────────────────────────────────────────────────────
# Chothia CDR 길이 정상 범위
CDR_LENGTH_NORMAL = {
    "H1": (5, 8),  "H2": (5, 8),  "H3": (3, 20),
    "L1": (6, 12), "L2": (3, 7),  "L3": (7, 11),
}

def analyze_cdrs(vh: str, vl: str) -> dict:
    """CDR 길이 및 서열 특성 분석
    CDR-H3 끝 위치는 VH 길이에서 FR4(WGQGTLVTVSS, 11 aa)를 뺀 값으로 동적 계산.
    """
    # CDR-H3 end = VH 길이 - FR4 길이 (트라스투주맙 FR4 = WGQGTLVTVSS, 11 aa)
    VH_FR4_LEN = 11
    h3_end_actual = len(vh) - VH_FR4_LEN
    vh_cdr_actual = {**VH_CDR, "H3": (VH_CDR["H3"][0], h3_end_actual)}

    cdrs = {}
    for name, (start, end) in vh_cdr_actual.items():
        seq = vh[start - 1: end]
        lo, hi = CDR_LENGTH_NORMAL[name]
        cdrs[name] = {
            "sequence": seq,
            "length":   len(seq),
            "normal_range": f"{lo}-{hi}",
            "length_ok": lo <= len(seq) <= hi,
        }
    for name, (start, end) in VL_CDR.items():
        seq = vl[start - 1: end]
        lo, hi = CDR_LENGTH_NORMAL[name]
        cdrs[name] = {
            "sequence": seq,
            "length":   len(seq),
            "normal_range": f"{lo}-{hi}",
            "length_ok": lo <= len(seq) <= hi,
        }
    abnormal = [n for n, v in cdrs.items() if not v["length_ok"]]
    return {"cdrs": cdrs, "abnormal_length": abnormal}


# ─── 5. 용해도 예측 (CamSol-intrinsic 간소화) ────────────────────────────────
# 원본 논문: Sormanni et al. 2015
CAMSOL_WEIGHT = {
    "R": 1.0, "K": 0.7, "D": 0.6, "E": 0.6, "N": 0.3, "Q": 0.3,
    "S": 0.2, "T": 0.2, "G": 0.1, "H": 0.1, "A": -0.1,
    "V": -0.6, "I": -0.8, "L": -0.8, "M": -0.4,
    "F": -1.0, "W": -1.5, "Y": -0.8, "C": -0.5, "P": 0.0,
}

def calc_solubility(seq: str) -> dict:
    """CamSol-intrinsic 단순화 버전 — 양수 = 가용성 양호"""
    score = sum(CAMSOL_WEIGHT.get(aa, 0) for aa in seq) / len(seq)
    return {
        "camSol_score": round(score, 3),
        "solubility":   "GOOD"   if score > 0.2  else
                        "OK"     if score > 0.0   else
                        "RISK",
    }


# ─── 6. 인간성 점수 (BioPhi OASis) ───────────────────────────────────────────
def calc_humanness_biophi(vh: str, vl: str) -> dict:
    """BioPhi OASis 인간성 점수 (설치 시)"""
    try:
        from biophi.humanization.methods.humanness import (
            OASisHumannessMethod
        )
        method = OASisHumannessMethod()
        vh_score = method.get_sequence_humanness(vh, chain_type="H")
        vl_score = method.get_sequence_humanness(vl, chain_type="L")
        return {
            "vh_oasis": round(float(vh_score), 3),
            "vl_oasis": round(float(vl_score), 3),
            "avg_oasis": round((float(vh_score) + float(vl_score)) / 2, 3),
            "humanness": "HIGH" if min(vh_score, vl_score) > 0.8 else
                         "OK"   if min(vh_score, vl_score) > 0.6 else "LOW",
        }
    except Exception as e:
        # BioPhi 사용 불가 → ANARCI 기반 단순 추정
        return _fallback_humanness(vh, vl, str(e))


def _fallback_humanness(vh: str, vl: str, reason: str = "") -> dict:
    """BioPhi 없을 때 — 게르민라인 유사도 기반 추정"""
    # 트라스투주맙 FR과의 유사도 (인간화된 항체이므로 기준점)
    ref_vh = ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARI"
               "YPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYY")
    ref_vl = ("DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYS"
               "ASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYC")
    l = min(len(vh), len(ref_vh))
    vh_id = sum(a == b for a, b in zip(vh[:l], ref_vh[:l])) / l
    l = min(len(vl), len(ref_vl))
    vl_id = sum(a == b for a, b in zip(vl[:l], ref_vl[:l])) / l
    return {
        "vh_germline_sim": round(vh_id, 3),
        "vl_germline_sim": round(vl_id, 3),
        "humanness":       "HIGH" if min(vh_id, vl_id) > 0.8 else
                           "OK"   if min(vh_id, vl_id) > 0.6 else "LOW",
        "method":          "germline_similarity",
        "note":            reason[:80] if reason else "",
    }


# ─── 7. 종합 개발가능성 점수 ─────────────────────────────────────────────────
def compute_developability_score(results: dict) -> dict:
    """
    각 항목 점수 → 가중 합산 → 100점 만점
    임계점: ≥ 70 → 개발 권장, 50-70 → 조건부, < 50 → 재설계
    """
    score = 100.0
    flags = []

    # pI (7.5-9.5 최적 — CAR 발현 및 정제)
    pi = results["biophysical"]["pi"]
    if pi < 6.0 or pi > 10.5:
        score -= 15; flags.append(f"pI {pi} — 최적 범위(6-10.5) 벗어남")
    elif pi < 7.0 or pi > 9.5:
        score -= 5

    # GRAVY (< 0 선호)
    gravy = results["biophysical"]["gravy"]
    if gravy > 0.5:
        score -= 15; flags.append(f"GRAVY {gravy} — 높은 소수성 (집합 위험)")
    elif gravy > 0.0:
        score -= 5

    # 불안정성 지수 (< 40 안정)
    if not results["biophysical"]["stable"]:
        score -= 10; flags.append("불안정성 지수 ≥ 40")

    # 화학적 취약부위 (CDR 내 HIGH 위험)
    high_risks = [l for l in results["liabilities"] if l["risk"] == "HIGH"]
    score -= min(len(high_risks) * 8, 30)
    if high_risks:
        types = list(set(l["type"] for l in high_risks))
        flags.append(f"CDR 취약부위 {len(high_risks)}개: {', '.join(types)}")

    # 유리 Cys
    for chain_key in ("vh_cysteine", "vl_cysteine"):
        if results.get(chain_key, {}).get("unpaired_cys"):
            score -= 20; flags.append(f"{chain_key} 유리 Cys 존재")

    # 집합 위험
    agg = results["aggregation"]
    if agg["aggregation_risk"] == "HIGH":
        score -= 15; flags.append("CDR 소수성/양전하 패치 다수")
    elif agg["aggregation_risk"] == "MEDIUM":
        score -= 7

    # CDR 길이 이상
    if results["cdr_profile"]["abnormal_length"]:
        score -= 5 * len(results["cdr_profile"]["abnormal_length"])
        flags.append(f"CDR 비정상 길이: {results['cdr_profile']['abnormal_length']}")

    # 용해도
    if results["solubility"]["solubility"] == "RISK":
        score -= 12; flags.append("용해도 위험 (CamSol < 0)")
    elif results["solubility"]["solubility"] == "OK":
        score -= 3

    # 인간성
    hum = results["humanness"].get("humanness", "OK")
    if hum == "LOW":
        score -= 15; flags.append("낮은 인간성 점수 (면역원성 위험)")
    elif hum == "OK":
        score -= 5

    score = max(0.0, min(100.0, score))
    level = ("EXCELLENT (개발 권장)" if score >= 80 else
             "GOOD (소규모 최적화)" if score >= 70 else
             "MODERATE (개선 필요)" if score >= 50 else
             "POOR (재설계 권장)")

    return {
        "score":          round(score, 1),
        "level":          level,
        "flags":          flags,
        "recommendation": flags if flags else ["개발가능성 우수 — 다음 단계 진행 가능"],
    }


# ─── 전체 분석 ────────────────────────────────────────────────────────────────
def run_developability(vh: str, vl: str) -> dict:
    scfv = vh + "GGGGSGGGGSGGGGS" + vl

    # CDR-H3 끝 위치를 VH 길이에서 동적 계산 (FR4 = 11 aa)
    VH_FR4_LEN = 11
    vh_cdr_dynamic = {**VH_CDR, "H3": (VH_CDR["H3"][0], len(vh) - VH_FR4_LEN)}
    vh_cdr_pos = get_cdr_positions(len(vh), vh_cdr_dynamic)
    vl_cdr_pos = get_cdr_positions(len(vl), VL_CDR)

    print("[1/7] 물리화학적 특성 계산...")
    biopysical_vh = calc_biophysical(vh)
    biopysical_vl = calc_biophysical(vl)
    biopysical_sc = calc_biophysical(scfv)

    print("[2/7] 화학적 취약부위 탐지...")
    liabilities = (find_liabilities(vh, vh_cdr_pos, "VH") +
                   find_liabilities(vl, vl_cdr_pos, "VL"))

    print("[3/7] Cys 분석...")
    vh_cys = check_cysteine(vh, "VH")
    vl_cys = check_cysteine(vl, "VL")

    print("[4/7] 집합 위험 분석...")
    vh_agg = calc_aggregation_risk(vh, vh_cdr_pos)
    vl_agg = calc_aggregation_risk(vl, vl_cdr_pos)
    agg_combined = {
        "hydrophobic_patches": vh_agg["hydrophobic_patches"] + vl_agg["hydrophobic_patches"],
        "positive_patches":    vh_agg["positive_patches"]    + vl_agg["positive_patches"],
        "n_hydro_patches":     vh_agg["n_hydro_patches"]     + vl_agg["n_hydro_patches"],
        "n_pos_patches":       vh_agg["n_pos_patches"]        + vl_agg["n_pos_patches"],
        "aggregation_risk":    ("HIGH"   if "HIGH"   in (vh_agg["aggregation_risk"], vl_agg["aggregation_risk"]) else
                                "MEDIUM" if "MEDIUM" in (vh_agg["aggregation_risk"], vl_agg["aggregation_risk"]) else
                                "LOW"),
    }

    print("[5/7] CDR 프로파일...")
    cdr_profile = analyze_cdrs(vh, vl)

    print("[6/7] 용해도 예측...")
    solubility = calc_solubility(scfv)

    print("[7/7] 인간성 점수 (BioPhi/germline)...")
    humanness = calc_humanness_biophi(vh, vl)

    result = {
        "vh": vh,
        "vl": vl,
        "scfv_len": len(scfv),
        "biophysical":  biopysical_sc,
        "biophysical_vh": biopysical_vh,
        "biophysical_vl": biopysical_vl,
        "liabilities":  liabilities,
        "vh_cysteine":  vh_cys,
        "vl_cysteine":  vl_cys,
        "aggregation":  agg_combined,
        "cdr_profile":  cdr_profile,
        "solubility":   solubility,
        "humanness":    humanness,
    }

    result["developability"] = compute_developability_score(result)
    return result


# ─── 보고서 출력 ─────────────────────────────────────────────────────────────
RISK_ICON = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "OK": "✅", "RISK": "🔴",
             "GOOD": "✅", "EXCELLENT": "⭐", "POOR": "❌"}

def print_report(r: dict):
    dev = r["developability"]
    bio = r["biophysical"]

    print()
    print("=" * 65)
    print("   항체 scFv 개발가능성(Developability) 평가 보고서")
    print("=" * 65)

    # 종합 점수
    sc = dev["score"]
    icon = ("⭐" if sc >= 80 else "✅" if sc >= 70 else "⚠️" if sc >= 50 else "❌")
    print(f"\n  {icon} 종합 점수: {sc}/100 — {dev['level']}")
    print()

    # 물리화학적 특성
    print("[ 물리화학적 특성 (scFv 전체) ]")
    print(f"  분자량  : {bio['mw_kda']:.1f} kDa")
    print(f"  pI      : {bio['pi']}  {'✅' if 6 <= bio['pi'] <= 10.5 else '⚠️'}")
    print(f"  순전하  : {bio['net_charge_ph74']:+.1f} (pH 7.4)")
    print(f"  GRAVY   : {bio['gravy']}  {'✅' if bio['gravy'] < 0 else '⚠️'}")
    print(f"  불안정성: {bio['instability_index']}  {'✅ 안정' if bio['stable'] else '⚠️ 불안정'}")
    sol = r["solubility"]
    print(f"  용해도  : CamSol {sol['camSol_score']:+.3f} — {sol['solubility']}")

    # CDR 프로파일
    print()
    print("[ CDR 프로파일 ]")
    for name, info in r["cdr_profile"]["cdrs"].items():
        ok = "✅" if info["length_ok"] else "⚠️"
        print(f"  {ok} {name}: {info['sequence']} ({info['length']} aa, 정상 {info['normal_range']})")

    # 화학적 취약부위
    print()
    high = [l for l in r["liabilities"] if l["risk"] == "HIGH"]
    med  = [l for l in r["liabilities"] if l["risk"] == "MEDIUM"]
    print(f"[ 화학적 취약부위 ] HIGH:{len(high)} MEDIUM:{len(med)}")
    for l in sorted(r["liabilities"], key=lambda x: {"HIGH":0,"MEDIUM":1,"LOW":2}[x["risk"]])[:10]:
        icon2 = RISK_ICON.get(l["risk"], "")
        cdr_tag = " ← CDR" if l["in_cdr"] else ""
        print(f"  {icon2} {l['type']:18s} {l['chain']} pos{l['position']:3d} "
              f"'{l['motif']}'{cdr_tag}")

    # Cys
    print()
    print("[ Cys 분석 ]")
    for chain, cys in [("VH", r["vh_cysteine"]), ("VL", r["vl_cysteine"])]:
        u = cys["unpaired_cys"]
        icon3 = "🔴" if u else "✅"
        print(f"  {icon3} {chain}: 전체 {cys['total_cys']}개"
              f"  보존 이황화결합 {cys['conserved_disulfide']}"
              f"  유리 Cys {u if u else '없음'}")

    # 집합
    print()
    agg = r["aggregation"]
    icon4 = RISK_ICON.get(agg["aggregation_risk"], "")
    print(f"[ 집합 위험 ] {icon4} {agg['aggregation_risk']}")
    print(f"  소수성 패치(CDR): {agg['n_hydro_patches']}개  "
          f"양전하 패치(CDR): {agg['n_pos_patches']}개")

    # 인간성
    print()
    hum = r["humanness"]
    hum_icon = {"HIGH": "✅", "OK": "🟡", "LOW": "🔴"}.get(hum["humanness"], "")
    print(f"[ 인간성 ] {hum_icon} {hum['humanness']}")
    if "vh_oasis" in hum:
        print(f"  OASis VH: {hum['vh_oasis']:.3f}  VL: {hum['vl_oasis']:.3f}")
    else:
        print(f"  Germline 유사도 VH: {hum.get('vh_germline_sim', '-'):.3f}  "
              f"VL: {hum.get('vl_germline_sim', '-'):.3f}")

    # 개선 권고
    print()
    print("[ 개선 권고 ]")
    for flag in dev["flags"]:
        print(f"  ⚠️  {flag}")
    if not dev["flags"]:
        print("  ✅ 문제 없음 — 다음 단계 진행 권장")

    print("=" * 65)


# ─── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="항체 scFv 개발가능성 평가"
    )
    parser.add_argument("--vh",  required=True, help="VH 아미노산 서열")
    parser.add_argument("--vl",  required=True, help="VL 아미노산 서열")
    parser.add_argument("--out", default=None,  help="결과 JSON 저장 경로")
    parser.add_argument("--fasta", default=None, help="VH/VL FASTA 파일 (--vh/--vl 대체)")
    args = parser.parse_args()

    # FASTA 입력 지원
    if args.fasta:
        from Bio import SeqIO
        seqs = {r.id: str(r.seq) for r in SeqIO.parse(args.fasta, "fasta")}
        vh_key = next((k for k in seqs if "VH" in k.upper()), None)
        vl_key = next((k for k in seqs if "VL" in k.upper()), None)
        vh = seqs[vh_key] if vh_key else args.vh
        vl = seqs[vl_key] if vl_key else args.vl
    else:
        vh = args.vh.strip().upper()
        vl = args.vl.strip().upper()

    print(f"VH ({len(vh)} aa): {vh[:30]}...")
    print(f"VL ({len(vl)} aa): {vl[:30]}...")

    result = run_developability(vh, vl)
    print_report(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # JSON 직렬화 (numpy 타입 처리)
        def to_python(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, dict): return {k: to_python(v) for k, v in obj.items()}
            if isinstance(obj, list): return [to_python(v) for v in obj]
            return obj
        with open(out_path, "w") as f:
            json.dump(to_python(result), f, indent=2, ensure_ascii=False)
        print(f"\n[저장] {out_path}")


if __name__ == "__main__":
    main()
