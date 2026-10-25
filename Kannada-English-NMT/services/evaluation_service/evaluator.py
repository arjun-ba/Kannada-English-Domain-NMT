"""
Microservice: Evaluation Service
Calculates BLEU scores (BLEU-1 to BLEU-4), brevity penalty, n-gram precision, and test set benchmarks.
"""

import math
from collections import Counter
from typing import Dict, List, Any, Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class EvaluationService:
    """Computes NLP evaluation metrics for Machine Translation."""

    def __init__(self):
        self.smoother = SmoothingFunction().method1

    def calculate_bleu(self, hypothesis: str, reference: str) -> Dict[str, Any]:
        """
        Calculates BLEU scores (BLEU-1, 2, 3, 4, cumulative) and details.
        """
        hyp_tokens = hypothesis.strip().lower().split()
        ref_tokens = reference.strip().lower().split()

        if not hyp_tokens or not ref_tokens:
            return {
                "bleu_cumulative": 0.0,
                "bleu_1": 0.0,
                "bleu_2": 0.0,
                "bleu_3": 0.0,
                "bleu_4": 0.0,
                "brevity_penalty": 0.0,
                "hypothesis_length": len(hyp_tokens),
                "reference_length": len(ref_tokens),
                "ngram_precisions": {}
            }

        # NLTK BLEU scores
        b1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smoother)
        b2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoother)
        b3 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoother)
        b4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoother)

        # Brevity penalty
        c = len(hyp_tokens)
        r = len(ref_tokens)
        bp = 1.0 if c > r else math.exp(1 - (r / c)) if c > 0 else 0.0

        # N-gram precision breakdown
        precisions = {}
        for n in range(1, 5):
            hyp_ngrams = [tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1)]
            ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
            hyp_counts = Counter(hyp_ngrams)
            ref_counts = Counter(ref_ngrams)

            clipped_matches = sum(min(count, ref_counts[ng]) for ng, count in hyp_counts.items())
            total = len(hyp_ngrams)
            precisions[f"{n}-gram"] = {
                "matched": clipped_matches,
                "total": total,
                "precision": round((clipped_matches / total) * 100, 2) if total > 0 else 0.0
            }

        return {
            "bleu_cumulative": round(b4 * 100, 2),
            "bleu_1": round(b1 * 100, 2),
            "bleu_2": round(b2 * 100, 2),
            "bleu_3": round(b3 * 100, 2),
            "bleu_4": round(b4 * 100, 2),
            "brevity_penalty": round(bp, 3),
            "hypothesis_length": c,
            "reference_length": r,
            "ngram_precisions": precisions
        }

    def evaluate_test_set(self, pairs: List[Dict[str, str]]) -> Dict[str, Any]:
        """Runs evaluation over a collection of sample sentence pairs."""
        if not pairs:
            return {"avg_bleu": 0.0, "total_evaluated": 0}

        scores = []
        for pair in pairs:
            hyp = pair.get("translated", "")
            ref = pair.get("reference", "")
            res = self.calculate_bleu(hyp, ref)
            scores.append(res["bleu_cumulative"])

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "avg_bleu": round(avg_score, 2),
            "total_evaluated": len(pairs),
            "individual_scores": scores
        }
