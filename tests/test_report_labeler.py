"""
Unit tests for src/report_labeler.py
"""

import pandas as pd
import numpy as np
from src.report_labeler import (
    label_report,
    calibrate_soft_labels,
    ReportLabeler,
    FINDINGS,
    STATE_POSITIVE,
    STATE_NEGATIVE,
    STATE_UNMENTIONED,
)


def test_label_report_english():
    text = (
        "1. Trace joint effusion. "
        "2. Anterior cruciate ligament is intact. "
        "3. Full-thickness tear of the medial meniscus. "
        "4. No evidence of fracture."
    )
    res = label_report(text)

    # 1. Unpacking check: state, weight
    acl_state, acl_weight = res["ACL"]
    assert acl_state == STATE_NEGATIVE, f"Expected explicit_negative, got {acl_state}"
    assert acl_weight == 1.0, f"Expected weight 1.0, got {acl_weight}"
    assert res["ACL"].soft_target == 0.05

    # 2. Trace/mild qualifier downgrading
    eff_state, eff_weight = res["Effusion"]
    assert eff_state == STATE_POSITIVE, f"Expected explicit_positive, got {eff_state}"
    assert eff_weight == 1.0
    assert abs(res["Effusion"].soft_target - 0.35) < 1e-5, f"Expected 0.35, got {res['Effusion'].soft_target}"

    # 3. Full-thickness/severe qualifier
    mm_state, mm_weight = res["Medial Meniscus"]
    assert mm_state == STATE_POSITIVE
    assert mm_weight == 1.0
    assert abs(res["Medial Meniscus"].soft_target - 0.85) < 1e-5

    # 4. Explicit negative
    fx_state, fx_weight = res["Fracture"]
    assert fx_state == STATE_NEGATIVE
    assert fx_weight == 1.0
    assert res["Fracture"].soft_target == 0.05

    # 5. Unmentioned
    mcl_state, mcl_weight = res["MCL"]
    assert mcl_state == STATE_UNMENTIONED
    assert mcl_weight == 0.3
    assert res["MCL"].soft_target == 0.15


def test_multilingual():
    # Spanish
    text_es = "Sin derrame articular. Rotura completa del ligamento cruzado anterior."
    res_es = label_report(text_es)
    assert res_es["Effusion"].state == STATE_NEGATIVE
    assert res_es["ACL"].state == STATE_POSITIVE
    assert abs(res_es["ACL"].soft_target - 0.85) < 1e-5

    # French
    text_fr = "Discret epanchement intra-articulaire. Menisque medial intact."
    res_fr = label_report(text_fr)
    assert res_fr["Effusion"].state == STATE_POSITIVE
    assert abs(res_fr["Effusion"].soft_target - 0.35) < 1e-5
    assert res_fr["Medial Meniscus"].state == STATE_NEGATIVE

    # German
    text_de = "Grosser Gelenkerguss. Vorderes Kreuzband intakt. Knochenkontusion im medialen Tibiaplateau."
    res_de = label_report(text_de)
    assert res_de["Effusion"].state == STATE_POSITIVE
    assert abs(res_de["Effusion"].soft_target - 0.85) < 1e-5
    assert res_de["ACL"].state == STATE_NEGATIVE
    assert res_de["Contusion"].state == STATE_POSITIVE

    # Portuguese
    text_pt = "Sem sinais de fratura. Pequeno cisto de Baker."
    res_pt = label_report(text_pt)
    assert res_pt["Fracture"].state == STATE_NEGATIVE
    assert res_pt["Baker's Cyst"].state == STATE_POSITIVE
    assert abs(res_pt["Baker's Cyst"].soft_target - 0.35) < 1e-5


def test_calibration():
    # Create synthetic reports dataframe
    labeler = ReportLabeler()
    sample_reports = [
        "Normal knee. No effusion. Intact ACL.",
        "Large joint effusion with ACL tear.",
        "Mild joint effusion, medial meniscus tear.",
        "Fracture of tibia. Bone contusion.",
        "Unremarkable exam.",
    ] * 12  # 60 studies

    df_reports = pd.DataFrame({
        "StudyInstanceUID": [f"study_{i}" for i in range(len(sample_reports))],
        "report": sample_reports,
    })

    # Create synthetic gold dataframe for 58 studies
    np.random.seed(42)
    df_gold = pd.DataFrame({"StudyInstanceUID": [f"study_{i}" for i in range(58)]})
    for f in FINDINGS:
        df_gold[f] = np.random.binomial(1, 0.25, size=58)

    calibrated_df = calibrate_soft_labels(df_reports, df_gold)
    cal_map = calibrated_df.attrs["calibration_map"]

    for f in FINDINGS:
        p_neg = cal_map[f][STATE_NEGATIVE]
        p_unm = cal_map[f][STATE_UNMENTIONED]
        p_pos = cal_map[f][STATE_POSITIVE]

        # Verify monotonic ordering: pos > unm > neg with 0.05 margin
        assert p_unm >= p_neg + 0.05 - 1e-5, f"{f}: unm ({p_unm}) < neg ({p_neg}) + 0.05"
        assert p_pos >= p_unm + 0.05 - 1e-5, f"{f}: pos ({p_pos}) < unm ({p_unm}) + 0.05"


if __name__ == "__main__":
    test_label_report_english()
    test_multilingual()
    test_calibration()
    print("ALL TESTS PASSED SUCCESSFULLY!")
