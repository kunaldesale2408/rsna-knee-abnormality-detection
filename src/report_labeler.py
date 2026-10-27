"""
RSNA Knee Abnormality Detection — Radiology Report Labeler.

Converts multilingual radiology report text into calibrated soft probability targets
and confidence weights for 12 binary findings.
Supports English, Spanish, French, German, Portuguese, and other clinical text.
"""

from typing import Dict, Tuple, List, Optional, Any, Union
import re
import unicodedata
import numpy as np
import pandas as pd


FINDINGS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# State constants
STATE_POSITIVE = "explicit_positive"
STATE_NEGATIVE = "explicit_negative"
STATE_UNMENTIONED = "unmentioned"

# Qualifier constants
QUAL_NONE = "none"
QUAL_MILD = "mild"
QUAL_SEVERE = "severe"


class FindingLabel(tuple):
    """
    Label output for a single finding.
    Can be unpacked as a 2-tuple (state, confidence_weight) for spec compliance:
        state, weight = result["ACL"]
    Or accessed via properties:
        .state: "explicit_positive", "explicit_negative", "unmentioned"
        .confidence_weight / .weight: confidence weight (float)
        .soft_target: calibrated soft probability target (float)
        .qualifier: "mild", "severe", "none"
    """
    def __new__(
        cls,
        state: str,
        confidence_weight: float,
        soft_target: float,
        qualifier: str = QUAL_NONE,
    ):
        instance = super(FindingLabel, cls).__new__(cls, (state, float(confidence_weight)))
        instance.state = state
        instance.confidence_weight = float(confidence_weight)
        instance.weight = float(confidence_weight)
        instance.soft_target = float(soft_target)
        instance.qualifier = qualifier
        return instance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "weight": self.confidence_weight,
            "soft_target": self.soft_target,
            "qualifier": self.qualifier,
        }

    def __repr__(self) -> str:
        return (
            f"FindingLabel(state='{self.state}', "
            f"confidence_weight={self.confidence_weight:.2f}, "
            f"soft_target={self.soft_target:.2f}, "
            f"qualifier='{self.qualifier}')"
        )


def normalize_text(text: str) -> str:
    """
    Normalize text: strip diacritics / accents, lowercase, and clean whitespace.
    """
    if not isinstance(text, str):
        return ""
    # NFKD decomposition separates base letters from accent marks
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Replace German ß with ss
    cleaned = stripped.replace("ß", "ss").lower()
    return cleaned


# Regex patterns across English, Spanish, French, German, Portuguese
FINDING_PATTERNS = {
    "ACL": re.compile(
        r"\b(anterior\s+cruciate(\s+ligament)?|"
        r"acl|a\.c\.l|"
        r"cruzado\s+anterior|"
        r"lca|l\.c\.a|"
        r"croise\s+anterieur|"
        r"vordere[snr]?\s+kreuzband|"
        r"vkb|v\.k\.b)\b",
        re.IGNORECASE,
    ),
    "MCL": re.compile(
        r"\b(medial\s+collateral(\s+ligament)?|"
        r"mcl|m\.c\.l|"
        r"colateral\s+medial|colateral\s+interno|"
        r"lcm|l\.c\.m|"
        r"collateral\s+medial|collateral\s+interne|"
        r"innenband|mediale[snr]?\s+kollateralband)\b",
        re.IGNORECASE,
    ),
    "Medial Meniscus": re.compile(
        r"\b(medial\s+menisc(us|al)|"
        r"menisco\s+(medial|interno)|"
        r"menisque\s+(medial|interne)|"
        r"innenmeniskus|mediale[snr]?\s+meniskus)\b",
        re.IGNORECASE,
    ),
    "Lateral Meniscus": re.compile(
        r"\b(lateral\s+menisc(us|al)|"
        r"menisco\s+(lateral|externo)|"
        r"menisque\s+(lateral|externe)|"
        r"aussenmeniskus|laterale[snr]?\s+meniskus)\b",
        re.IGNORECASE,
    ),
    "Medial OA": re.compile(
        r"\b(medial\s+(compartment|tibiofemoral|joint\s+space\s+narrowing|osteoarthritis|gonarthrosis)|"
        r"compartimento\s+(medial|interno)|femorotibial\s+(medial|interno)|"
        r"pinzamiento\s+(medial|interno)|artrosis\s+(medial|interna)|gonartrosis\s+medial|"
        r"compartiment\s+(medial|interne)|femoro-?tibial\s+(medial|interne)|"
        r"pincement\s+(medial|interne)|arthrose\s+(mediale|interne)|"
        r"mediale[snr]?\s+(kompartiment|gelenkspalt|gonarthrose|arthrose))\b",
        re.IGNORECASE,
    ),
    "Lateral OA": re.compile(
        r"\b(lateral\s+(compartment|tibiofemoral|joint\s+space\s+narrowing|osteoarthritis|gonarthrosis)|"
        r"compartimento\s+(lateral|externo)|femorotibial\s+(lateral|externo)|"
        r"pinzamiento\s+(lateral|externo)|artrosis\s+(lateral|externa)|gonartrosis\s+lateral|"
        r"compartiment\s+(lateral|externe)|femoro-?tibial\s+(lateral|externe)|"
        r"pincement\s+(lateral|externe)|arthrose\s+(laterale|externe)|"
        r"laterale[snr]?\s+(kompartiment|gelenkspalt|gonarthrose|arthrose))\b",
        re.IGNORECASE,
    ),
    "PF OA": re.compile(
        r"\b(patellofemoral|pf\s+compartment|patellar\s+cartilage|patellofemoral\s+osteoarthritis|"
        r"chondromalacia\s+patell(a|ae)|patella\s+cartilage|"
        r"femoropatelar|compartimento\s+femoropatelar|cartilago\s+rotuliano|"
        r"artrosis\s+patelofemoral|condromalacia\s+rotuliana|"
        r"femoro-?patellaire|cartilage\s+patellaire|arthrose\s+femoro-?patellaire|"
        r"chondromalacie\s+rotulienne|femoropatellargelenk|retropatellare\s+arthrose|"
        r"retropatellarer\s+knorpel|chondropathia\s+patellae|cartilagem\s+patelar)\b",
        re.IGNORECASE,
    ),
    "Effusion": re.compile(
        r"\b(effusion|joint\s+fluid|knee\s+fluid|hydrarthrosis|intra-?articular\s+fluid|"
        r"derrame(\s+articular)?|liquido\s+articular|hidrartrosis|"
        r"epanchement(\s+intra-?articulaire)?|liquide\s+articulaire|hydarthrose|"
        r"gelenkerguss|erguss|intraartikulare\s+flussigkeit|flussigkeitsansammlung|"
        r"hidrartrose)\b",
        re.IGNORECASE,
    ),
    "Synovitis": re.compile(
        r"\b(synovitis|synovial\s+(thickening|enhancement|hypertrophy)|"
        r"sinovitis|engrosamiento\s+sinovial|hipertrofia\s+sinovial|"
        r"synovite|epaississement\s+synovial|hypertrophie\s+synoviale|"
        r"synovitischer\s+reizzustand|synoviale\s+verdickung|espessamento\s+sinovial)\b",
        re.IGNORECASE,
    ),
    "Baker's": re.compile(
        r"\b(baker(\'?s)?(\s+cyst)?|popliteal\s+cyst|posterior\s+cyst|"
        r"gastrocnemius-?semimembranosus(\s+bursa)?|"
        r"quiste\s+(de\s+baker|popliteo|sinovial\s+popliteo)|"
        r"kyste\s+(de\s+baker|poplite)|"
        r"baker-?zyste|poplitealzyste|"
        r"cisto\s+(de\s+baker|popliteo))\b",
        re.IGNORECASE,
    ),
    "Baker's Cyst": re.compile(
        r"\b(baker(\'?s)?(\s+cyst)?|popliteal\s+cyst|posterior\s+cyst|"
        r"gastrocnemius-?semimembranosus(\s+bursa)?|"
        r"quiste\s+(de\s+baker|popliteo|sinovial\s+popliteo)|"
        r"kyste\s+(de\s+baker|poplite)|"
        r"baker-?zyste|poplitealzyste|"
        r"cisto\s+(de\s+baker|popliteo))\b",
        re.IGNORECASE,
    ),
    "Contusion": re.compile(
        r"\b(contusion|bone\s+bruise|bone\s+marrow\s+edema|bme|trabecular\s+microfracture|"
        r"edema\s+oseo|edema\s+de\s+medula\s+osea|contusion\s+osea|"
        r"oedeme\s+osseux|edeme\s+osseux|contusion\s+osseuse|"
        r"knochenkontusion|knochenmarkodem|trabekulare\s+mikrofraktur|"
        r"contusao|edema\s+osseo|edema\s+da\s+medula\s+ossea)\b",
        re.IGNORECASE,
    ),
    "Fracture": re.compile(
        r"\b(fracture|fracture\s+line|insufficiency\s+fracture|cortical\s+disruption|"
        r"fractura|trazo\s+de\s+fractura|fractura\s+por\s+insuficiencia|"
        r"trait\s+de\s+fracture|fracture\s+de\s+fatigue|fracture\s+par\s+insuffisance|"
        r"fraktur|frakturlinie|insuffizienzfraktur|knochenbruch|"
        r"fratura|traco\s+de\s+fratura)\b",
        re.IGNORECASE,
    ),
}

# Negation terms across English, Spanish, French, German, Portuguese
NEGATION_PATTERNS = [
    re.compile(r"\b(no|not|without|absence\s+of|free\s+of|negative\s+for|rules?\s+out|no\s+evidence\s+of|no\s+sign\s+of|no\s+definite|no\s+acute)\b", re.IGNORECASE),
    re.compile(r"\b(sin|no\s+se\s+observa|no\s+hay|ausencia\s+de|sin\s+evidencia\s+de|sin\s+signos\s+de|descartar|sin\s+hallazgos)\b", re.IGNORECASE),
    re.compile(r"\b(pas\s+de|aucun|aucune|absence\s+de|sans|sans\s+anomalie|sans\s+signe\s+de|pas\s+d\'anomalie)\b", re.IGNORECASE),
    re.compile(r"\b(kein|keine|keinen|keinerlei|ohne|kein\s+nachweis|ohne\s+befund|frei\s+von|ohne\s+fassbare)\b", re.IGNORECASE),
    re.compile(r"\b(sem|ausencia\s+de|nao\s+ha|nao\s+se\s+observa|sem\s+sinais\s+de|sem\s+evidencia\s+de)\b", re.IGNORECASE),
]

# Normal/intact terms (used to classify mentions as negative)
INTACT_PATTERNS = re.compile(
    r"\b(intact|normal|unremarkable|preserved|in-tact|continuous|uninjured|"
    r"intacto|intacta|conservado|conservada|sin\s+alteraciones|"
    r"respecte|respectee|sans\s+anomalie|sans\s+particularite|"
    r"intakt|regelrecht|unauffallig|ohne\s+befund|ohne\s+pathologischer\s+befund|"
    r"preservado|preservada)\b",
    re.IGNORECASE,
)

# Abnormality indicators for anatomical structures (ligaments, menisci, compartments)
ABNORMALITY_PATTERNS = re.compile(
    r"\b(tear|torn|rupture|disruption|lesion|sprain|fraying|extrusion|maceration|"
    r"attenuation|defect|gap|abnormal|edema|narrowing|narrowed|sclerosis|osteophyte|"
    r"loss|degeneration|chondromalacia|"
    r"rotura|desgarro|lesion|fissure|esguince|pinzamiento|artrosis|desgaste|"
    r"dechirure|rupture|fissure|entorse|pincement|arthrose|"
    r"riss|ruptur|lasion|teilruptur|zerrung|schadigung|verschleiss|abnutzung|"
    r"estiramento|laceracao|pincamento|artrose)\b",
    re.IGNORECASE,
)

# Severity qualifiers: mild / small
QUALIFIER_MILD = re.compile(
    r"\b(small|mild|trace|minimal|slight|tiny|scant|faint|subtle|low-?grade|grade\s+(1|i)\b|"
    r"pequeno|pequena|leve|minimo|minima|escaso|escasa|discreto|discreta|grado\s+(1|i)\b|"
    r"petit|petite|minime|discret|discrete|leger|legere|faible|"
    r"klein|kleine|kleiner|gering|geringe|geringer|mild|milde|minimal|minimale|spuren|spurenweise|dezent|grad\s+(1|i)\b|"
    r"grau\s+(1|i)\b)",
    re.IGNORECASE,
)

# Severity qualifiers: severe / large / complete
QUALIFIER_SEVERE = re.compile(
    r"\b(large|complete|full-?thickness|severe|massive|marked|high-?grade|grade\s+(3|iii)\b|extensive|gross|"
    r"grande|completo|completa|espesor\s+completo|severo|severa|grave|masivo|masiva|marcado|marcada|alto\s+grado|grado\s+(3|iii)\b|"
    r"grand|grande|complet|complete|transfixiant|transfixiante|severe|massif|massive|marque|marquee|etendu|etendue|"
    r"gross|grosse|grosser|vollstandig|komplett|schwer|massiv|ausgepragt|hochgradig|grad\s+(3|iii)\b|"
    r"espessura\s+total|grau\s+(3|iii)\b)",
    re.IGNORECASE,
)


class ReportLabeler:
    """
    Regex-based multilingual report labeler for 12 knee MRI findings.
    """
    def __init__(self):
        self.findings = list(FINDINGS)
        self.finding_patterns = FINDING_PATTERNS
        self.negation_patterns = NEGATION_PATTERNS
        self.intact_pattern = INTACT_PATTERNS
        self.abnormality_pattern = ABNORMALITY_PATTERNS
        self.qualifier_mild = QUALIFIER_MILD
        self.qualifier_severe = QUALIFIER_SEVERE

    def _split_clauses(self, text: str) -> List[str]:
        """
        Split report text into sentence/clause units.
        """
        # Split on sentence boundaries, list numbers, colons, newlines, and semicolons
        parts = re.split(r"[.\n;:]|\b\d+[\.\)]\s*", text)
        return [p.strip() for p in parts if p.strip()]

    def _has_negation(self, clause: str) -> bool:
        """
        Check if clause contains any negation indicator.
        """
        for neg_pat in self.negation_patterns:
            if neg_pat.search(clause):
                return True
        return False

    def _detect_qualifier(self, clause: str) -> str:
        """
        Determine severity qualifier in the clause.
        """
        if self.qualifier_severe.search(clause):
            return QUAL_SEVERE
        if self.qualifier_mild.search(clause):
            return QUAL_MILD
        return QUAL_NONE

    def label_report(self, text: str) -> Dict[str, FindingLabel]:
        """
        Convert raw report text into a dictionary {finding: (state, confidence_weight)}.
        Returns FindingLabel instances for all 12 findings.
        """
        norm_text = normalize_text(text)
        clauses = self._split_clauses(norm_text)

        results: Dict[str, FindingLabel] = {}

        for finding in self.findings:
            pattern = self.finding_patterns[finding]

            matched_clauses = [c for c in clauses if pattern.search(c)]

            if not matched_clauses:
                # Finding is not mentioned in report
                results[finding] = FindingLabel(
                    state=STATE_UNMENTIONED,
                    confidence_weight=0.3,
                    soft_target=0.15,
                    qualifier=QUAL_NONE,
                )
                continue

            # Evaluate each mention of the finding
            finding_state = STATE_UNMENTIONED
            finding_qualifier = QUAL_NONE

            for clause in matched_clauses:
                is_negated = self._has_negation(clause)
                is_intact = bool(self.intact_pattern.search(clause))
                is_abnormal = bool(self.abnormality_pattern.search(clause))

                # Primary pathology findings where mention implies presence unless negated
                is_pathology_entity = finding in {
                    "Effusion",
                    "Synovitis",
                    "Baker's",
                    "Baker's Cyst",
                    "Contusion",
                    "Fracture",
                }

                if is_negated or is_intact:
                    # Explicit negative: negated or described as intact/normal
                    # If previously found positive in another clause, keep positive
                    if finding_state != STATE_POSITIVE:
                        finding_state = STATE_NEGATIVE
                elif is_pathology_entity or is_abnormal:
                    # Explicit positive
                    finding_state = STATE_POSITIVE
                    qual = self._detect_qualifier(clause)
                    if qual == QUAL_SEVERE or (qual == QUAL_MILD and finding_qualifier != QUAL_SEVERE):
                        finding_qualifier = qual
                else:
                    # Anatomical structure mentioned without explicit pathology or negation
                    if finding_state == STATE_UNMENTIONED:
                        finding_state = STATE_NEGATIVE

            # Map state and qualifier to target and weight
            if finding_state == STATE_POSITIVE:
                confidence_weight = 1.0
                if finding_qualifier == QUAL_MILD:
                    soft_target = 0.35
                else:
                    # Default explicit positive or severe qualifier
                    soft_target = 0.85
            elif finding_state == STATE_NEGATIVE:
                confidence_weight = 1.0
                soft_target = 0.05
            else:
                confidence_weight = 0.3
                soft_target = 0.15

            results[finding] = FindingLabel(
                state=finding_state,
                confidence_weight=confidence_weight,
                soft_target=soft_target,
                qualifier=finding_qualifier,
            )

        # Ensure both Baker's and Baker's Cyst aliases are populated for downstream compatibility
        if "Baker's" in results and "Baker's Cyst" not in results:
            results["Baker's Cyst"] = results["Baker's"]
        elif "Baker's Cyst" in results and "Baker's" not in results:
            results["Baker's"] = results["Baker's Cyst"]

        return results

    def label_dataframe(
        self,
        df: pd.DataFrame,
        report_col: str = "report",
        study_id_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Label an entire DataFrame of reports, returning a DataFrame with columns:
        <Finding>_target, <Finding>_weight, <Finding>_state, <Finding>_qualifier.
        """
        if report_col not in df.columns:
            # Auto-detect alternative report text column names
            candidates = [c for c in df.columns if any(k in c.lower() for k in ["report", "text", "impression"])]
            if candidates:
                report_col = candidates[0]
            else:
                raise ValueError(f"Report column '{report_col}' not found in DataFrame.")

        rows = []
        for idx, row in df.iterrows():
            text = str(row[report_col]) if pd.notna(row[report_col]) else ""
            labels = self.label_report(text)

            study_id = row.get(study_id_col, idx) if study_id_col else row.get("StudyInstanceUID", idx)
            out_row: Dict[str, Any] = {"StudyInstanceUID": study_id}

            for finding, lbl in labels.items():
                out_row[f"{finding}_target"] = lbl.soft_target
                out_row[f"{finding}_weight"] = lbl.confidence_weight
                out_row[f"{finding}_state"] = lbl.state
                out_row[f"{finding}_qualifier"] = lbl.qualifier
                # Also provide direct finding column name matching target
                out_row[finding] = lbl.soft_target

            rows.append(out_row)

        return pd.DataFrame(rows)


# Module-level convenience functions
_DEFAULT_LABELER = ReportLabeler()


def label_report(text: str) -> Dict[str, FindingLabel]:
    """
    Module-level function to label a single raw report text string.
    Returns dict {finding: (state, confidence_weight)}.
    """
    return _DEFAULT_LABELER.label_report(text)


def calibrate_soft_labels(
    df_reports: pd.DataFrame,
    df_gold_58: pd.DataFrame,
    study_id_col: str = "StudyInstanceUID",
    report_col: str = "report",
    min_support: int = 5,
    margin: float = 0.05,
    shrinkage_m: float = 5.0,
) -> pd.DataFrame:
    """
    Calibrate report-derived soft labels against 58 gold labels using empirical-Bayes.

    Requirements:
    - Computes empirical-Bayes calibration
    - Enforces monotonic ordering: positive_mean > unmentioned_mean > negative_mean (with 0.05 margin)
    - Falls back to hierarchical shrinkage pooled across all 12 labels when per-label support < 5

    Parameters:
    - df_reports: DataFrame containing reports for all studies
    - df_gold_58: DataFrame containing the 58 gold label studies (binary 0/1 columns for findings)
    - study_id_col: Column name identifying the study
    - min_support: Minimum sample count threshold before hierarchical shrinkage dominates (default: 5)
    - margin: Minimum margin between ordered state targets (default: 0.05)
    - shrinkage_m: Empirical-Bayes prior weight (default: 5.0)

    Returns:
    - pd.DataFrame: Calibrated DataFrame with updated soft targets and weights.
    """
    # 1. Ensure weak labels are extracted for df_reports
    state_cols_exist = all(f"{f}_state" in df_reports.columns for f in FINDINGS)
    if state_cols_exist:
        df_weak = df_reports.copy()
    else:
        labeler = ReportLabeler()
        df_weak = labeler.label_dataframe(df_reports, report_col=report_col, study_id_col=study_id_col)

    # 2. Identify overlapping gold studies
    if study_id_col not in df_weak.columns and "StudyInstanceUID" in df_weak.columns:
        study_id_col = "StudyInstanceUID"

    gold_studies = set(df_gold_58[study_id_col].astype(str))
    overlap_mask = df_weak[study_id_col].astype(str).isin(gold_studies)
    df_overlap_weak = df_weak[overlap_mask].copy()

    # Align gold labels
    gold_merged = pd.merge(
        df_overlap_weak[[study_id_col] + [f"{f}_state" for f in FINDINGS]],
        df_gold_58[[study_id_col] + [f for f in FINDINGS if f in df_gold_58.columns]],
        on=study_id_col,
        suffixes=("_weak", "_gold"),
    )

    # 3. Compute global pooled mean across all 12 labels for each state
    pooled_counts = {
        STATE_POSITIVE: {"total": 0, "pos": 0, "default_prior": 0.85},
        STATE_UNMENTIONED: {"total": 0, "pos": 0, "default_prior": 0.15},
        STATE_NEGATIVE: {"total": 0, "pos": 0, "default_prior": 0.05},
    }

    for finding in FINDINGS:
        state_col = f"{finding}_state"
        gold_col = finding if finding in gold_merged.columns else f"{finding}_gold"
        if state_col not in gold_merged.columns or gold_col not in gold_merged.columns:
            continue

        for state in [STATE_POSITIVE, STATE_UNMENTIONED, STATE_NEGATIVE]:
            mask = gold_merged[state_col] == state
            count = int(mask.sum())
            if count > 0:
                pos = float(gold_merged.loc[mask, gold_col].sum())
                pooled_counts[state]["total"] += count
                pooled_counts[state]["pos"] += pos

    pooled_priors = {}
    for state, info in pooled_counts.items():
        if info["total"] > 0:
            pooled_priors[state] = info["pos"] / info["total"]
        else:
            pooled_priors[state] = info["default_prior"]

    # 4. Per-finding empirical-Bayes calibration with hierarchical shrinkage
    calibrated_map: Dict[str, Dict[str, float]] = {}

    for finding in FINDINGS:
        state_col = f"{finding}_state"
        gold_col = finding if finding in gold_merged.columns else f"{finding}_gold"

        calibrated_map[finding] = {}

        raw_means = {}
        supports = {}

        for state in [STATE_NEGATIVE, STATE_UNMENTIONED, STATE_POSITIVE]:
            prior = pooled_priors[state]
            if state_col in gold_merged.columns and gold_col in gold_merged.columns:
                mask = gold_merged[state_col] == state
                n = int(mask.sum())
                k = float(gold_merged.loc[mask, gold_col].sum()) if n > 0 else 0.0
            else:
                n = 0
                k = 0.0

            supports[state] = n
            # Empirical Bayes shrinkage: (k + m * prior) / (n + m)
            shrunk_mean = (k + shrinkage_m * prior) / (n + shrinkage_m)
            raw_means[state] = shrunk_mean

        # 5. Enforce monotonic ordering:
        # positive_mean > unmentioned_mean > negative_mean (with margin)
        neg_val = float(np.clip(raw_means[STATE_NEGATIVE], 0.01, 0.40))
        unm_val = float(raw_means[STATE_UNMENTIONED])
        pos_val = float(raw_means[STATE_POSITIVE])

        # Enforce unmentioned >= negative + margin
        if unm_val < neg_val + margin:
            unm_val = neg_val + margin

        # Enforce positive >= unmentioned + margin
        if pos_val < unm_val + margin:
            pos_val = unm_val + margin

        # Clip within [0.01, 0.99]
        neg_val = float(np.clip(neg_val, 0.01, 0.99))
        unm_val = float(np.clip(unm_val, 0.01, 0.99))
        pos_val = float(np.clip(pos_val, 0.01, 0.99))

        calibrated_map[finding][STATE_NEGATIVE] = neg_val
        calibrated_map[finding][STATE_UNMENTIONED] = unm_val
        calibrated_map[finding][STATE_POSITIVE] = pos_val

    # 6. Apply calibrated values to df_weak
    df_calibrated = df_weak.copy()
    for finding in FINDINGS:
        state_col = f"{finding}_state"
        target_col = f"{finding}_target"
        qual_col = f"{finding}_qualifier"

        if state_col not in df_calibrated.columns:
            continue

        cal_neg = calibrated_map[finding][STATE_NEGATIVE]
        cal_unm = calibrated_map[finding][STATE_UNMENTIONED]
        cal_pos = calibrated_map[finding][STATE_POSITIVE]

        # Calculate mild target: interpolates between unmentioned and positive
        cal_mild = unm_val_target = float(cal_unm + 0.4 * (cal_pos - cal_unm))

        new_targets = []
        for _, row in df_calibrated.iterrows():
            st = row[state_col]
            ql = row.get(qual_col, QUAL_NONE)
            if st == STATE_POSITIVE:
                if ql == QUAL_MILD:
                    new_targets.append(cal_mild)
                else:
                    new_targets.append(cal_pos)
            elif st == STATE_NEGATIVE:
                new_targets.append(cal_neg)
            else:
                new_targets.append(cal_unm)

        df_calibrated[target_col] = new_targets
        df_calibrated[finding] = new_targets

    df_calibrated.attrs["calibration_map"] = calibrated_map
    df_calibrated.attrs["pooled_priors"] = pooled_priors

    return df_calibrated
