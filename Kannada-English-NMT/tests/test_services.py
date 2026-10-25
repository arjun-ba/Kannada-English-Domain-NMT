"""
Unit and Integration Tests for Microservices Suite
Tests Preprocessing, Dataset, Translation, and Evaluation independently.
"""

import os
import unittest
import torch

from services.preprocessing_service.normalizer import TextNormalizer, AGRI_DOMAIN_LEXICON
from services.preprocessing_service.tokenizer import Vocabulary, SOS_IDX, EOS_IDX
from services.dataset_service.data_loader import DatasetManager
from services.translation_service.attention import BahdanauAttention
from services.translation_service.seq2seq_model import Encoder, Decoder, Seq2Seq
from services.translation_service.translator import TranslationEngine
from services.evaluation_service.evaluator import EvaluationService

class TestMicroservices(unittest.TestCase):

    def test_01_preprocessing_normalizer(self):
        """Test Unicode normalization and lexicon extraction."""
        raw_kn = "ರೈತರು   ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ   ಮಾಡಿಸಬೇಕು."
        clean_kn = TextNormalizer.normalize_kannada(raw_kn)
        self.assertNotIn("   ", clean_kn)
        
        domain_terms = TextNormalizer.detect_domain_terms(clean_kn, "kannada")
        self.assertTrue(len(domain_terms) > 0)
        self.assertEqual(domain_terms[0]["source_term"], "ರೈತರು")

    def test_02_preprocessing_tokenizer(self):
        """Test vocabulary indexing and encode/decode."""
        vocab = Vocabulary("kannada_test")
        vocab.add_sentence("ರೈತರು ಬೆಳೆಗಳಿಗೆ ನೀರು ಹಾಕಬೇಕು")
        
        indices = vocab.encode("ರೈತರು ನೀರು", add_special_tokens=True)
        self.assertEqual(indices[0], SOS_IDX)
        self.assertEqual(indices[-1], EOS_IDX)
        
        decoded = vocab.decode(indices, remove_special_tokens=True)
        self.assertEqual(decoded, "ರೈತರು ನೀರು")

    def test_03_dataset_service(self):
        """Test data loading and splitting."""
        mgr = DatasetManager()
        summary = mgr.load_data()
        self.assertGreater(summary["total_samples"], 0)
        self.assertGreater(summary["kannada_vocab_size"], 10)
        self.assertGreater(summary["english_vocab_size"], 10)

    def test_04_translation_attention_module(self):
        """Test Bahdanau Attention tensor shapes."""
        hidden_dim = 16
        batch_size = 2
        src_len = 5
        
        attention = BahdanauAttention(hidden_dim)
        hidden = torch.randn(batch_size, hidden_dim)
        encoder_outputs = torch.randn(src_len, batch_size, hidden_dim * 2)
        
        context, weights = attention(hidden, encoder_outputs)
        self.assertEqual(context.shape, (batch_size, hidden_dim * 2))
        self.assertEqual(weights.shape, (batch_size, src_len))

    def test_05_translation_engine(self):
        """Test end-to-end translation pipeline."""
        mgr = DatasetManager()
        engine = TranslationEngine(mgr)
        
        res = engine.translate("ರೈತರು ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಮಾಡಿಸಬೇಕು.", direction="kn->en")
        self.assertIn("translated_text", res)
        self.assertIn("attention_matrix", res)
        self.assertIn("confidence", res)
        self.assertGreater(len(res["translated_text"]), 0)

    def test_06_evaluation_service(self):
        """Test BLEU metric calculation."""
        evaluator = EvaluationService()
        hyp = "Farmers should get the soil tested."
        ref = "Farmers should get the soil tested."
        
        scores = evaluator.calculate_bleu(hyp, ref)
        self.assertEqual(scores["bleu_cumulative"], 100.0)
        self.assertEqual(scores["bleu_1"], 100.0)

if __name__ == "__main__":
    unittest.main()
