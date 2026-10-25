"""
Microservice: Dataset Service
Handles data loading, domain categorization, train/validation splitting, and dataset statistics.
"""

import os
import pandas as pd
from typing import Dict, List, Tuple, Any
from ..preprocessing_service.normalizer import TextNormalizer
from ..preprocessing_service.tokenizer import Vocabulary

class DatasetManager:
    """Manages the bilingual parallel dataset for Kannada-English Domain NMT."""

    def __init__(self, csv_path: str = None):
        if csv_path is None:
            # Default to the data directory relative to this module
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.csv_path = os.path.abspath(os.path.join(current_dir, "..", "..", "data", "dataset.csv"))
        else:
            self.csv_path = csv_path

        self.df: pd.DataFrame = pd.DataFrame()
        self.kannada_vocab = Vocabulary("kannada")
        self.english_vocab = Vocabulary("english")
        self.loaded: bool = False
        self.train_pairs: List[Tuple[str, str, str]] = []
        self.val_pairs: List[Tuple[str, str, str]] = []

    def load_data(self) -> Dict[str, Any]:
        """Loads and processes the dataset."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset file not found at {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        # Ensure column names match
        columns = [c.lower().strip() for c in self.df.columns]
        self.df.columns = columns

        # Normalize texts
        self.df['kannada_clean'] = self.df['kannada'].apply(TextNormalizer.normalize_kannada)
        self.df['english_clean'] = self.df['english'].apply(TextNormalizer.normalize_english)

        # Build Vocabularies
        self.kannada_vocab = Vocabulary("kannada")
        self.english_vocab = Vocabulary("english")

        pairs = []
        for _, row in self.df.iterrows():
            kn = row['kannada_clean']
            en = row['english_clean']
            topic = row.get('topic', 'Agriculture')
            self.kannada_vocab.add_sentence(kn)
            self.english_vocab.add_sentence(en)
            pairs.append((kn, en, topic))

        # Train / Validation split (80/20)
        split_idx = max(1, int(len(pairs) * 0.8))
        self.train_pairs = pairs[:split_idx]
        self.val_pairs = pairs[split_idx:] if len(pairs) > split_idx else pairs[:split_idx]
        self.loaded = True

        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Returns statistical overview of the dataset."""
        if not self.loaded:
            self.load_data()

        topic_counts = self.df['topic'].value_counts().to_dict() if 'topic' in self.df.columns else {}

        return {
            "total_samples": len(self.df),
            "train_samples": len(self.train_pairs),
            "val_samples": len(self.val_pairs),
            "kannada_vocab_size": self.kannada_vocab.num_words,
            "english_vocab_size": self.english_vocab.num_words,
            "domain_topics": topic_counts,
            "primary_domain": "Agriculture & Rural Science"
        }

    def get_all_samples(self) -> List[Dict[str, Any]]:
        """Returns list of all dataset records with domain metadata."""
        if not self.loaded:
            self.load_data()
        
        samples = []
        for idx, row in self.df.iterrows():
            samples.append({
                "id": idx + 1,
                "kannada": row['kannada'],
                "english": row['english'],
                "topic": row.get('topic', 'General Agriculture'),
                "kannada_tokens": len(str(row['kannada']).split()),
                "english_tokens": len(str(row['english']).split())
            })
        return samples

    def add_custom_sentence(self, kannada: str, english: str, topic: str = "Custom Agriculture") -> Dict[str, Any]:
        """Allows dynamic addition of domain sentence pairs without restarting services."""
        kn_clean = TextNormalizer.normalize_kannada(kannada)
        en_clean = TextNormalizer.normalize_english(english)
        
        # Add to vocabs
        self.kannada_vocab.add_sentence(kn_clean)
        self.english_vocab.add_sentence(en_clean)
        
        new_row = pd.DataFrame([{
            'kannada': kannada,
            'english': english,
            'topic': topic,
            'kannada_clean': kn_clean,
            'english_clean': en_clean
        }])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.train_pairs.append((kn_clean, en_clean, topic))
        
        # Save to disk
        self.df[['kannada', 'english', 'topic']].to_csv(self.csv_path, index=False)
        return self.get_summary()
