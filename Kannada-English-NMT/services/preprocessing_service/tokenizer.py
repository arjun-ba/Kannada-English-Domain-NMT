"""
Microservice: Preprocessing Service - Tokenizer
Vocabulary management, sequence encoding/decoding, padding, and special token indexing.
"""

import json
import os
from typing import List, Dict, Tuple, Optional

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

class Vocabulary:
    """Manages word-to-index and index-to-word mappings for NMT."""

    def __init__(self, name: str):
        self.name = name
        self.word2index: Dict[str, int] = {}
        self.word2count: Dict[str, int] = {}
        self.index2word: Dict[int, str] = {
            PAD_IDX: PAD_TOKEN,
            SOS_IDX: SOS_TOKEN,
            EOS_IDX: EOS_TOKEN,
            UNK_IDX: UNK_TOKEN
        }
        self.num_words: int = 4  # Start with special tokens

    def add_sentence(self, sentence: str) -> None:
        """Tokenize sentence and add words to vocabulary."""
        for word in sentence.strip().split():
            self.add_word(word)

    def add_word(self, word: str) -> None:
        """Add individual word to vocabulary."""
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def encode(self, sentence: str, add_special_tokens: bool = True) -> List[int]:
        """Convert a sentence string to list of token IDs."""
        tokens = sentence.strip().split()
        indices = []
        if add_special_tokens:
            indices.append(SOS_IDX)
        for token in tokens:
            indices.append(self.word2index.get(token, UNK_IDX))
        if add_special_tokens:
            indices.append(EOS_IDX)
        return indices

    def decode(self, indices: List[int], remove_special_tokens: bool = True) -> str:
        """Convert list of token IDs back into string."""
        words = []
        for idx in indices:
            if remove_special_tokens and idx in (PAD_IDX, SOS_IDX, EOS_IDX):
                continue
            words.append(self.index2word.get(idx, UNK_TOKEN))
        return " ".join(words)

    def to_dict(self) -> Dict:
        """Serialize vocabulary state to dict."""
        return {
            "name": self.name,
            "num_words": self.num_words,
            "word2index": self.word2index,
            "word2count": self.word2count,
            "index2word": {str(k): v for k, v in self.index2word.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Vocabulary':
        """Load vocabulary from serialized dict."""
        vocab = cls(data["name"])
        vocab.num_words = data["num_words"]
        vocab.word2index = data["word2index"]
        vocab.word2count = data.get("word2count", {})
        vocab.index2word = {int(k): v for k, v in data["index2word"].items()}
        return vocab

    def save(self, filepath: str) -> None:
        """Save vocabulary to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """Load vocabulary from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
