"""
Interactive CLI Demo: Kannada-English Domain NMT
Allows testing translations, attention scores, and BLEU metrics directly in the terminal.
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Ensure UTF-8 console output on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from services.dataset_service.data_loader import DatasetManager
from services.translation_service.translator import TranslationEngine
from services.evaluation_service.evaluator import EvaluationService

def main():
    print("\n" + "="*65)
    print(" 🌾 Kannada-English Domain NMT - Terminal CLI Demo")
    print("="*65)
    
    print("\n[1/3] Initializing Dataset Service...")
    dataset_mgr = DatasetManager()
    summary = dataset_mgr.load_data()
    print(f"      Loaded {summary['total_samples']} domain samples across topics: {', '.join(summary['domain_topics'].keys())}")
    
    print("\n[2/3] Initializing Translation Engine & Training Seq2Seq GRU...")
    engine = TranslationEngine(dataset_mgr)
    print("      Neural Model Ready! (Bahdanau Attention Enabled)")
    
    print("\n[3/3] Initializing Evaluation Service...")
    evaluator = EvaluationService()
    print("      NLTK BLEU Metric Suite Ready!")
    
    test_sentences = [
        "ರೈತರು ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಮಾಡಿಸಬೇಕು.",
        "ಬೆಳೆಗಳಿಗೆ ಸರಿಯಾದ ಸಮಯದಲ್ಲಿ ನೀರು ಹಾಕಬೇಕು.",
        "ಸಸ್ಯಗಳಿಗೆ ಕೀಟಗಳ ದಾಳಿ ಉಂಟಾಗಬಹುದು.",
        "ಕೃಷಿಯಲ್ಲಿ ಸಾವಯವ ಗೊಬ್ಬರವನ್ನು ಬಳಸಬಹುದು."
    ]
    
    print("\n" + "-"*65)
    print(" LIVE TRANSLATION & EVALUATION SAMPLES:")
    print("-"*65)
    
    for idx, kn in enumerate(test_sentences, 1):
        res = engine.translate(kn, direction="kn->en")
        print(f"\nSample #{idx}:")
        print(f"  Source (KN)   : {kn}")
        print(f"  Output (EN)   : {res['translated_text']}")
        print(f"  Confidence    : {res['confidence']}%")
        print(f"  Method        : {res['method']}")
        if res.get('domain_terms'):
            terms = [f"{t['source_term']} -> {t['target_meaning']}" for t in res['domain_terms']]
            print(f"  Domain Terms  : {', '.join(terms)}")
            
    print("\n" + "="*65)
    print(" CLI Demo completed successfully!")
    print(" To launch the Interactive Web Studio, run: python run_server.py")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
