#!/usr/bin/env python3
"""
파라토프-에피토프 상보성 분석 + CDR-특이적 면역원성 분석
- FAP Blade 6-7 에피토프: E311(-), D313(-), R356(+), K360(+), F358(hydro)
- CDR 전하/소수성 profile vs 에피토프 상보성 점수 계산
- MHC-II CDR-specific 15-mer 스캔 (FR 플랭크 포함)
- 최종 종합 스코어카드 생성

Input:  fap_design/candidates/top5_final.json
Output: fap_design/analysis/paratope_analysis.json
        fap_design/analysis/final_scorecard.json
"""

import json, os, math

# ── 아미노산 성질 ──────────────────────────────────────────────────────────────
CHARGE_POS = set("KRH")   # 양전하
CHARGE_NEG = set("DE")    # 음전하
HYDROPHOBIC = set("VILMFYW")
AROMATIC    = set("FYW")
POLAR       = set("STNQ")
SPECIAL     = set("CGP")

# ── FAP Blade 6-7 에피토프 키 잔기 (전하/성질) ─────────────────────────────
FAP_EPITOPE = {
    "E311": {"charge": -1, "type": "neg"},
    "D313": {"charge": -1, "type": "neg"},
    "R356": {"charge": +1, "type": "pos"},
    "K360": {"charge": +1, "type": "pos"},
    "F358": {"charge":  0, "type": "hydro"},
}
# 상보성: FAP 음전하 → CDR 양전하 필요; FAP 양전하 → CDR 음전하 필요; FAP 소수성 → CDR 방향족/소수성
COMPLEMENTARITY = {
    "neg": "pos",   # FAP 음전하 ↔ CDR 양전하
    "pos": "neg",   # FAP 양전하 ↔ CDR 음전하
    "hydro": "hydro",
}

# ── FR 플랭크 서열 (에피토프 컨텍스트 창 확장용) ─────────────────────────────
# Chothia VH CDR 위치: H1(26-32), H2(52-56), H3(97-~)
# 각 CDR 앞뒤 7aa FR 플랭크를 포함한 15-mer 스캔
VH_FR = {
    "H1_pre":  "CAAS",     # FR1 끝 4aa
    "H1_post": "YIHWV",    # FR2 앞 5aa
    "H2_pre":  "WVARIP",   # FR2 끝 6aa → H2 앞
    "H2_post": "YTRYA",    # FR3 앞 5aa
    "H3_pre":  "TAVYYC",   # FR3 끝 6aa (EDTAVYYC → C 포함)
    "H3_post": "WGQGT",    # FR4 앞 5aa
}

MHC2_WINDOW   = 15
MHC2_THRESHOLD_MOD  = 5
MHC2_THRESHOLD_HIGH = 7


def aa_type(aa):
    if aa in CHARGE_POS:  return "pos"
    if aa in CHARGE_NEG:  return "neg"
    if aa in HYDROPHOBIC: return "hydro"
    if aa in AROMATIC:    return "arom"
    if aa in POLAR:       return "polar"
    return "other"


def cdr_profile(seq):
    """CDR 서열의 전하/소수성 profile."""
    n_pos   = sum(1 for a in seq if a in CHARGE_POS)
    n_neg   = sum(1 for a in seq if a in CHARGE_NEG)
    n_hydro = sum(1 for a in seq if a in HYDROPHOBIC)
    n_arom  = sum(1 for a in seq if a in AROMATIC)
    n_polar = sum(1 for a in seq if a in POLAR)
    net_charge = n_pos - n_neg
    length = len(seq)
    return {
        "len": length,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_hydro": n_hydro,
        "n_arom": n_arom,
        "n_polar": n_polar,
        "net_charge": net_charge,
        "frac_hydro": round(n_hydro / length, 3) if length else 0,
    }


def complementarity_score(cdrs):
    """
    FAP 에피토프 대비 CDR 상보성 점수 (0-10).
    H3이 Blade 6-7과 가장 직접 접촉한다고 가정.
    E311/D313(neg) → H3 양전하 필요
    R356/K360(pos) → H3 음전하 or 극성 필요
    F358(hydro)    → H3 방향족/소수성 필요
    """
    h3 = cdrs.get("H3", "")
    score = 0.0
    notes = []

    # E311/D313 → H3 양전하 (K,R,H)
    n_pos_h3 = sum(1 for a in h3 if a in CHARGE_POS)
    if n_pos_h3 >= 2:
        score += 2.5; notes.append(f"H3 K/R/H×{n_pos_h3} ↔ E311/D313 ✅")
    elif n_pos_h3 == 1:
        score += 1.5; notes.append(f"H3 K/R/H×{n_pos_h3} ↔ E311/D313 ⚠️")
    else:
        notes.append("H3 양전하 없음 → E311/D313 상보성 부족 ❌")

    # R356/K360 → H3 음전하 or 극성 (D,E,S,T,N,Q)
    n_neg_h3 = sum(1 for a in h3 if a in CHARGE_NEG)
    n_polar_h3 = sum(1 for a in h3 if a in POLAR)
    if n_neg_h3 >= 1:
        score += 2.0; notes.append(f"H3 D/E×{n_neg_h3} ↔ R356/K360 ✅")
    elif n_polar_h3 >= 2:
        score += 1.0; notes.append(f"H3 극성×{n_polar_h3} ↔ R356/K360 ~중립")
    else:
        notes.append("H3 음전하/극성 부족 → R356/K360 반발 가능 ❌")

    # F358 → H3 방향족 (F,Y,W)
    n_arom_h3 = sum(1 for a in h3 if a in AROMATIC)
    if n_arom_h3 >= 3:
        score += 3.0; notes.append(f"H3 방향족×{n_arom_h3} ↔ F358 ✅✅")
    elif n_arom_h3 >= 1:
        score += 1.5; notes.append(f"H3 방향족×{n_arom_h3} ↔ F358 ✅")
    else:
        notes.append("H3 방향족 없음 → F358 스태킹 불가 ❌")

    # 보너스: H3 길이 (11-13aa 최적)
    h3_len = len(h3)
    if 11 <= h3_len <= 13:
        score += 1.5; notes.append(f"H3 길이 {h3_len}aa (최적 11-13) ✅")
    elif 8 <= h3_len <= 15:
        score += 0.5; notes.append(f"H3 길이 {h3_len}aa (수용 가능)")
    else:
        notes.append(f"H3 길이 {h3_len}aa (비최적)")

    # L2/L3 보너스 (보조 접촉)
    l3 = cdrs.get("L3", "")
    n_arom_l3 = sum(1 for a in l3 if a in AROMATIC)
    if n_arom_l3 >= 2:
        score += 1.0; notes.append(f"L3 방향족×{n_arom_l3} (보조 접촉) ✅")

    return {
        "score": round(min(score, 10.0), 2),
        "max": 10.0,
        "notes": notes,
        "grade": "A" if score >= 8 else "B" if score >= 6 else "C" if score >= 4 else "D",
    }


def mhc2_cdr_scan(cdrs, vhfr=None):
    """
    CDR+플랭크 컨텍스트로 15-mer MHC-II 스캔.
    훨씬 엄격: 전체 scFv 대신 CDR±7 플랭크만.
    """
    results = {}
    for cdr_key, cdr_seq in cdrs.items():
        if not cdr_seq:
            results[cdr_key] = {"hits": [], "risk": "N/A"}
            continue

        # 플랭크 추가
        if cdr_key == "H1":
            ctx = VH_FR["H1_pre"] + cdr_seq + VH_FR["H1_post"]
        elif cdr_key == "H2":
            ctx = VH_FR["H2_pre"] + cdr_seq + VH_FR["H2_post"]
        elif cdr_key == "H3":
            ctx = VH_FR["H3_pre"] + cdr_seq + VH_FR["H3_post"]
        else:
            # VL CDRs: 7aa 플랭크 없이 CDR만
            ctx = cdr_seq

        hits = []
        for i in range(max(1, len(ctx) - MHC2_WINDOW + 1)):
            pep = ctx[i:i + MHC2_WINDOW]
            if len(pep) < MHC2_WINDOW:
                continue
            n_hydro = sum(1 for aa in pep if aa in HYDROPHOBIC)
            if n_hydro >= MHC2_THRESHOLD_MOD:
                hits.append({
                    "peptide": pep,
                    "pos": i,
                    "hydrophobic_count": n_hydro,
                    "risk": "HIGH" if n_hydro >= MHC2_THRESHOLD_HIGH else "MOD",
                    "in_cdr": True,
                })

        n_high = sum(1 for h in hits if h["risk"] == "HIGH")
        n_mod  = sum(1 for h in hits if h["risk"] == "MOD")
        risk = "HIGH" if n_high >= 2 else ("MOD" if hits else "LOW")
        results[cdr_key] = {
            "context_len": len(ctx),
            "hits": hits,
            "n_high": n_high,
            "n_mod": n_mod,
            "risk": risk,
        }
    return results


def instability_index(seq):
    """Guruprasad's instability index (simplified)."""
    INSTAB = {
        "WW":1.0,"WC":1.0,"WM":24.68,"WH":24.68,"WY":1.0,
        "CW":1.0,"CC":1.0,"CM":33.6,"CH":33.6,"CY":1.0,
        "QW":1.0,"QC":1.0,"QM":1.0,"QH":24.68,"QY":1.0,
        "EW":1.0,"EC":1.0,"EM":1.0,"EH":-6.54,"EY":45.62,
        "GW":-7.49,"GC":-7.49,"GM":-7.49,"GH":-7.49,"GY":-7.49,
        "HW":-1.88,"HC":-1.88,"HM":-1.88,"HH":-1.88,"HY":-1.88,
        "IW":1.0,"IC":1.0,"IM":1.0,"IH":44.94,"IY":1.0,
        "KW":-7.49,"KC":-7.49,"KM":-7.49,"KH":-7.49,"KY":-7.49,
        "LW":24.68,"LC":24.68,"LM":1.0,"LH":1.0,"LY":1.0,
        "MW":1.0,"MC":1.0,"MM":1.0,"MH":58.28,"MY":44.94,
        "NW":1.0,"NC":1.0,"NM":1.0,"NH":1.0,"NY":1.0,
        "PW":1.0,"PC":1.0,"PM":1.0,"PH":-6.54,"PY":1.0,
        "RW":1.0,"RC":-6.54,"RM":1.0,"RH":20.26,"RY":1.0,
        "SW":1.0,"SC":33.6,"SM":1.0,"SH":1.0,"SY":1.0,
        "TW":1.0,"TC":1.0,"TM":1.0,"TH":1.0,"TY":1.0,
        "VW":1.0,"VC":1.0,"VM":1.0,"VH":1.0,"VY":-6.54,
        "YW":-7.49,"YC":1.0,"YM":44.94,"YH":13.34,"YY":13.34,
    }
    n = len(seq)
    if n < 2:
        return 0.0
    total = sum(INSTAB.get(seq[i:i+2], 1.0) for i in range(n-1))
    return round(10.0 / n * total, 2)


def gravy(seq):
    """Grand Average of Hydropathicity (Kyte-Doolittle)."""
    KD = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,
          "G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,
          "P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
    vals = [KD.get(a, 0) for a in seq]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def analyze_candidate(cand):
    cdrs  = cand["cdrs"]
    cid   = cand["id"]
    vh    = cand.get("vh", "")
    vl    = cand.get("vl", "")

    result = {"id": cid}

    # CDR profile
    result["cdr_profiles"] = {k: cdr_profile(v) for k, v in cdrs.items()}

    # Complementarity score
    result["complementarity"] = complementarity_score(cdrs)

    # CDR-specific MHC-II (with flanks)
    result["mhc2_cdr"] = mhc2_cdr_scan(cdrs)

    # 전체 MHC-II 위험 요약 (CDR-specific)
    cdr_risks = [v["risk"] for v in result["mhc2_cdr"].values() if v.get("risk") not in ("N/A",)]
    risk_rank = {"HIGH": 2, "MOD": 1, "LOW": 0}
    max_risk = max(cdr_risks, key=lambda r: risk_rank.get(r, 0)) if cdr_risks else "N/A"
    result["mhc2_cdr_overall"] = max_risk

    # Instability
    full_seq = vh + vl if vh and vl else ""
    result["instability_index"] = instability_index(full_seq) if full_seq else None
    result["gravy"]             = gravy(full_seq) if full_seq else None

    # ESM-2 from existing data
    result["esm2_pll"]   = cand.get("esm2_pll")
    result["dev_score"]  = cand.get("dev_score")
    result["pI"]         = cand.get("pI")
    result["camSol"]     = cand.get("camSol")
    result["humanness"]  = cand.get("humanness")
    result["patent_sim"] = cand.get("patent_max_sim")

    return result


def final_scorecard(analyzed_list):
    """
    7개 항목 → 100점 종합 스코어카드 (ColabFold/IgFold 전 단계 버전).
    1. ESM-2 PLL        (20점)
    2. 파라토프 상보성  (20점)
    3. 개발가능성 Dev   (15점)
    4. 면역원성 MHC-II  (15점) — CDR-specific
    5. 특허 안전성      (15점)
    6. pI 적정성        (10점)  6.5-9.5 최적
    7. CamSol 용해도    (5점)
    """
    scores = []
    for r in analyzed_list:
        s = {}
        s["id"] = r["id"]

        # 1. ESM-2 PLL (20점): top1(-0.2779) → 20점, 순위별 감점
        pll = r.get("esm2_pll")
        if pll is not None:
            # 범위 -0.2779 ~ -0.2800; normalize
            pll_score = 20 * max(0, min(1, (pll + 0.285) / 0.010))
        else:
            pll_score = 13.0  # 다양성 선택 → 중간 점수
        s["esm2_score"] = round(pll_score, 1)

        # 2. 파라토프 상보성 (20점)
        comp = r["complementarity"]["score"]
        s["complementarity_score"] = round(comp * 2.0, 1)  # /10 * 20

        # 3. 개발가능성 (15점): dev_score 100점 기준
        dev = r.get("dev_score") or 0
        s["dev_score_w"] = round(dev / 100 * 15, 1)

        # 4. 면역원성 MHC-II CDR-specific (15점): LOW=15, MOD=8, HIGH=0
        risk_map = {"LOW": 15, "MOD": 8, "HIGH": 0, "N/A": 8}
        s["immunogen_score"] = risk_map.get(r.get("mhc2_cdr_overall", "N/A"), 8)

        # 5. 특허 안전성 (15점): sim<50%→15, <55%→12, <60%→8, ≥60%→0
        sim = r.get("patent_sim") or 0
        s["patent_score"] = 15 if sim < 50 else 12 if sim < 55 else 8 if sim < 60 else 0

        # 6. pI 적정성 (10점): 6.5-9.5 최적 10점, ±1 8점, 그 외 감점
        pi = r.get("pI") or 7.0
        if 7.0 <= pi <= 9.0:
            s["pi_score"] = 10
        elif 6.5 <= pi <= 9.5:
            s["pi_score"] = 8
        elif 5.5 <= pi <= 10.0:
            s["pi_score"] = 5
        else:
            s["pi_score"] = 2

        # 7. CamSol (5점): > -0.05 → 5점, > -0.1 → 3점, ≤ -0.1 → 1점
        cam = r.get("camSol") or -0.1
        s["camSol_score"] = 5 if cam > -0.05 else 3 if cam > -0.1 else 1

        s["total"] = round(
            s["esm2_score"] + s["complementarity_score"] + s["dev_score_w"] +
            s["immunogen_score"] + s["patent_score"] + s["pi_score"] + s["camSol_score"], 1
        )
        scores.append(s)

    return sorted(scores, key=lambda x: -x["total"])


def main():
    in_path = "fap_design/candidates/top5_final.json"
    out_dir = "fap_design/analysis"
    os.makedirs(out_dir, exist_ok=True)

    with open(in_path) as f:
        top5 = json.load(f)

    # VH/VL 서열 재구성 (top5_final.json에 없으면 FASTA에서)
    # VH/VL은 FR+CDR 조합으로 재구성
    VH_FR1 = "EVQLVESGGGLVQPGGSLRLSCAAS"
    VH_FR2 = "YIHWVRQAPGKGLEWVARI"
    VH_FR3 = "YTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC"
    VH_FR4 = "WGQGTLVTVSS"
    VL_FR1 = "DIQMTQSPSSLSASVGDRVTITC"
    VL_FR2 = "WYQQKPGKAPKLLIY"
    VL_FR3 = "GVPSRFSGSRSGTDFTLTISSLQPEDFATYYC"
    VL_FR4 = "FGQGTKVEIK"
    LINKER = "GGGGSGGGGSGGGGSGGGGS"

    for cand in top5:
        cdrs = cand.get("cdrs", {})
        vh = (VH_FR1 + cdrs.get("H1","") + VH_FR2 + cdrs.get("H2","") +
              VH_FR3 + cdrs.get("H3","") + VH_FR4)
        vl = (VL_FR1 + cdrs.get("L1","") + VL_FR2 + cdrs.get("L2","") +
              VL_FR3 + cdrs.get("L3","") + VL_FR4)
        cand["vh"] = vh
        cand["vl"] = vl
        cand["scfv"] = vh + LINKER + vl

    # 분석
    print("=== 파라토프 상보성 분석 ===\n")
    analyzed = []
    for cand in top5:
        r = analyze_candidate(cand)
        analyzed.append(r)
        comp = r["complementarity"]
        print(f"[{r['id']}]")
        print(f"  상보성 점수: {comp['score']}/10 (등급 {comp['grade']})")
        for note in comp["notes"]:
            print(f"    • {note}")
        print(f"  MHC-II CDR-specific: {r['mhc2_cdr_overall']}")
        for cdr_key, cdr_res in r["mhc2_cdr"].items():
            if cdr_res.get("risk") not in ("LOW", "N/A"):
                print(f"    {cdr_key}: {cdr_res['risk']} ({cdr_res.get('n_high',0)} HIGH + {cdr_res.get('n_mod',0)} MOD hits)")
        print(f"  불안정 지수: {r['instability_index']}  GRAVY: {r['gravy']}")
        print()

    # 스코어카드
    scorecard = final_scorecard(analyzed)
    print("=== 최종 종합 스코어카드 (ColabFold 전 단계) ===")
    print(f"{'순위':<4} {'ID':<22} {'ESM2':>5} {'상보':>5} {'Dev':>5} {'Imm':>5} {'특허':>5} {'pI':>5} {'Cam':>5} {'합계':>6}")
    print("-" * 70)
    for i, s in enumerate(scorecard, 1):
        print(f"{i:<4} {s['id']:<22} {s['esm2_score']:>5} {s['complementarity_score']:>5} "
              f"{s['dev_score_w']:>5} {s['immunogen_score']:>5} {s['patent_score']:>5} "
              f"{s['pi_score']:>5} {s['camSol_score']:>5} {s['total']:>6}")

    # 저장
    analysis_path = os.path.join(out_dir, "paratope_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analyzed, f, ensure_ascii=False, indent=2)

    scorecard_path = os.path.join(out_dir, "final_scorecard.json")
    with open(scorecard_path, "w") as f:
        json.dump(scorecard, f, ensure_ascii=False, indent=2)

    print(f"\n[저장] {analysis_path}")
    print(f"[저장] {scorecard_path}")
    print("\n다음: bash fap_design/colabfold/run_colabfold.sh")


if __name__ == "__main__":
    main()
