"""
RSNA Knee Abnormality Detection — LLM-Enhanced Report Labeler (Phase 2).

Uses zero-shot Natural Language Inference (NLI) with multilingual mDeBERTa
("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli") to extract calibrated soft probability targets
for 12 binary knee abnormality findings from multilingual radiology reports.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.report_labeler import FINDINGS, normalize_text

FINDING_DESCRIPTIONS = {
    "ACL": "anterior cruciate ligament tear or abnormality",
    "MCL": "medial collateral ligament tear or abnormality",
    "Medial Meniscus": "medial meniscus tear or abnormality",
    "Lateral Meniscus": "lateral meniscus tear or abnormality",
    "Medial OA": "medial tibiofemoral osteoarthritis or joint space narrowing",
    "Lateral OA": "lateral tibiofemoral osteoarthritis or joint space narrowing",
    "PF OA": "patellofemoral osteoarthritis or cartilage loss",
    "Effusion": "knee joint effusion or fluid",
    "Synovitis": "synovitis or synovial thickening",
    "Baker's Cyst": "Baker cyst or popliteal cyst",
    "Contusion": "bone contusion or bone marrow edema",
    "Fracture": "fracture",
}


class LLMReportLabeler:
    """
    Zero-shot NLI multilingual labeler using mDeBERTa-v3-base-mnli-xnli.
    """
    def __init__(
        self,
        model_name_or_path: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 256,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.entailment_idx = 0
        self.contradiction_idx = 2
        self.neutral_idx = 1
        self._is_loaded = False

    def load_model(self) -> bool:
        """
        Load tokenizer and model from HuggingFace cache or local Kaggle dataset path.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            # Check candidate model paths
            candidate_paths = [
                self.model_name_or_path,
                "/kaggle/input/mdeberta-weights",
                "weights/mdeberta",
            ]
            loaded_path = self.model_name_or_path
            for p in candidate_paths:
                if os.path.isdir(p):
                    loaded_path = p
                    break

            print(f"[+] Loading NLI model from: {loaded_path} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(loaded_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(loaded_path)
            self.model.to(self.device)
            self.model.eval()

            # Identify entailment index from model config
            label2id = getattr(self.model.config, "label2id", {})
            for name, idx in label2id.items():
                nl = name.lower()
                if "entail" in nl:
                    self.entailment_idx = idx
                elif "contra" in nl:
                    self.contradiction_idx = idx
                elif "neut" in nl:
                    self.neutral_idx = idx

            self._is_loaded = True
            print(f"[+] Model loaded successfully. Entailment label index: {self.entailment_idx}")
            return True
        except Exception as e:
            print(f"[-] Warning: Failed to load mDeBERTa model ({e}). Fallback mode enabled.")
            self._is_loaded = False
            return False

    def predict_premise_hypothesis_batch(
        self,
        pairs: List[Tuple[str, str]],
    ) -> np.ndarray:
        """
        Predict entailment probabilities for a batch of (premise, hypothesis) pairs.
        Returns array of shape [len(pairs)] with entailment probabilities.
        """
        if not self._is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        all_probs = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            premises = [p[0] for p in batch]
            hypotheses = [p[1] for p in batch]

            inputs = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                entail_probs = probs[:, self.entailment_idx].cpu().numpy()

            all_probs.extend(entail_probs.tolist())

        return np.array(all_probs, dtype=np.float32)

    def label_report(self, report_text: str) -> Dict[str, Dict[str, float]]:
        """
        Label a single report text across all 12 findings using positive & negation NLI hypotheses.
        Returns:
            dict: {finding: {'soft_target': float, 'confidence_weight': float, 'p_pos': float, 'p_neg': float}}
        """
        norm_text = normalize_text(report_text)
        if not norm_text.strip():
            # Unmentioned defaults for empty reports
            return {
                f: {
                    "soft_target": 0.15,
                    "confidence_weight": 0.30,
                    "p_pos": 0.05,
                    "p_neg": 0.90,
                }
                for f in FINDINGS
            }

        pairs = []
        findings_order = list(FINDINGS)
        for finding in findings_order:
            desc = FINDING_DESCRIPTIONS[finding]
            pos_hypo = f"The MRI shows {desc}."
            neg_hypo = f"There is no evidence of {desc}."
            pairs.append((norm_text, pos_hypo))
            pairs.append((norm_text, neg_hypo))

        if self._is_loaded and self.model is not None:
            probs = self.predict_premise_hypothesis_batch(pairs)
        else:
            # Fallback estimation when model is not loaded locally
            probs = np.full(len(pairs), 0.10, dtype=np.float32)

        results = {}
        for idx, finding in enumerate(findings_order):
            p_pos = float(probs[2 * idx])
            p_neg = float(probs[2 * idx + 1])

            # Formula from spec:
            # Soft target = entailment_prob - 0.5 * negation_prob, clipped to [0.05, 0.95]
            soft_target = float(np.clip(p_pos - 0.5 * p_neg, 0.05, 0.95))

            # Confidence weight: certainty between positive and negative
            certainty = max(p_pos, p_neg)
            weight = float(np.clip(certainty * 1.2, 0.30, 1.00))

            results[finding] = {
                "soft_target": soft_target,
                "confidence_weight": weight,
                "p_pos": p_pos,
                "p_neg": p_neg,
            }

        return results

    def label_dataframe(
        self,
        df: pd.DataFrame,
        report_col: str = "report",
        study_id_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Label an entire DataFrame of reports, returning a DataFrame with:
        <Finding>_target, <Finding>_weight, <Finding>_p_pos, <Finding>_p_neg.
        """
        if report_col not in df.columns:
            for c in ["report", "report_text", "text", "impression"]:
                if c in df.columns:
                    report_col = c
                    break

        s_col = study_id_col or ("StudyInstanceUID" if "StudyInstanceUID" in df.columns else df.columns[0])

        rows = []
        for idx, row in df.iterrows():
            text = str(row.get(report_col, "")) if pd.notna(row.get(report_col)) else ""
            study_id = row.get(s_col, idx)
            lbls = self.label_report(text)

            out_row: Dict[str, Any] = {"StudyInstanceUID": study_id}
            for finding, vals in lbls.items():
                out_row[f"{finding}_target"] = vals["soft_target"]
                out_row[f"{finding}_weight"] = vals["confidence_weight"]
                out_row[f"{finding}_p_pos"] = vals["p_pos"]
                out_row[f"{finding}_p_neg"] = vals["p_neg"]
                out_row[finding] = vals["soft_target"]

            rows.append(out_row)

        return pd.DataFrame(rows)


def ensemble_weak_labels(
    regex_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    alpha: float = 0.5,
    study_id_col: str = "StudyInstanceUID",
) -> pd.DataFrame:
    """
    Ensemble soft labels: alpha * regex_soft + (1 - alpha) * llm_soft.
    Fallback when LLM does not achieve >= +0.03 margin over regex alone.
    """
    merged = pd.merge(
        regex_df,
        llm_df,
        on=study_id_col,
        suffixes=("_regex", "_llm"),
    )

    out_df = pd.DataFrame({study_id_col: merged[study_id_col]})

    for finding in FINDINGS:
        r_tgt = merged[f"{finding}_target_regex"] if f"{finding}_target_regex" in merged.columns else merged[f"{finding}_regex"]
        l_tgt = merged[f"{finding}_target_llm"] if f"{finding}_target_llm" in merged.columns else merged[f"{finding}_llm"]

        ens_tgt = alpha * r_tgt + (1.0 - alpha) * l_tgt
        out_df[f"{finding}_target"] = ens_tgt
        out_df[finding] = ens_tgt

        # Average confidence weights
        r_wt = merged.get(f"{finding}_weight_regex", 1.0)
        l_wt = merged.get(f"{finding}_weight_llm", 1.0)
        out_df[f"{finding}_weight"] = 0.5 * (r_wt + l_wt)

        # Retain state from regex for reference
        if f"{finding}_state_regex" in merged.columns:
            out_df[f"{finding}_state"] = merged[f"{finding}_state_regex"]

    return out_df
