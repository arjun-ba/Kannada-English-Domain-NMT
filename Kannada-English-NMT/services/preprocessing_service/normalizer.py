"""
Microservice: Preprocessing Service - Normalizer
Handles Unicode normalization, character cleaning, and domain keyword tagging for Kannada and English.
"""

import re
import unicodedata
from typing import Dict, List, Any

# Domain Specific Agriculture Lexicon mapping
AGRI_DOMAIN_LEXICON = {
    # Kannada to English domain entities
    "ರೈತರು": "farmers",
    "ರೈತ": "farmer",
    "ಮಣ್ಣಿನ": "soil",
    "ಮಣ್ಣು": "soil",
    "ಪರೀಕ್ಷೆ": "tested/test",
    "ಬೆಳೆಗಳಿಗೆ": "crops",
    "ಬೆಳೆಗಳು": "crops",
    "ಬೆಳೆ": "crop",
    "ನೀರು": "water",
    "ಸಸ್ಯಗಳಿಗೆ": "plants",
    "ಸಸ್ಯ": "plant",
    "ಕೀಟಗಳ": "pests",
    "ಕೀಟ": "pest",
    "ದಾಳಿ": "attack",
    "ಫಲವತ್ತತೆ": "fertility",
    "ಬೀಜಗಳನ್ನು": "seeds",
    "ಬೀಜ": "seed",
    "ಪೋಷಕಾಂಶಗಳನ್ನು": "nutrients",
    "ಪೋಷಕಾಂಶ": "nutrient",
    "ಗೊಬ್ಬರ": "fertilizer/manure",
    "ಸಾವಯವ": "organic",
    "ಕೀಟನಾಶಕ": "pesticide",
    "ಹವಾಮಾನ": "weather",
    "ನೀರಾವರಿ": "irrigation",
    "ಕಟಾವು": "harvest",
    "ಬಿತ್ತನೆ": "sowing",
    "ವಿಮೆ": "insurance",
    "ಕಳೆ": "weed",
    "ಉಳುಮೆ": "plowing",
    "ಕಾಂಪೋಸ್ಟ್": "compost"
}

class TextNormalizer:
    """Normalizes Kannada and English texts for NMT models."""

    @staticmethod
    def normalize_kannada(text: str) -> str:
        """
        Cleans and normalizes Kannada text:
        - Unicode normalization (NFC)
        - Removes redundant punctuation while preserving sentence delimiters
        - Fixes zero-width spaces / joins
        """
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize("NFC", text.strip())
        
        # Replace zero-width joiners / non-joiners if isolated
        text = text.replace("\u200b", "").replace("\ufeff", "")
        
        # Ensure spacing before/after punctuation
        text = re.sub(r'([.?!,;:\(\)\[\]"\'\u0964\u0965])', r' \1 ', text)
        
        # Condense multiple whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def normalize_english(text: str) -> str:
        """
        Cleans and normalizes English text:
        - Lowercasing
        - Standardizing punctuation spacing
        """
        if not text:
            return ""
        
        text = text.strip()
        # Add spacing around punctuation
        text = re.sub(r'([.?!,;:\(\)\[\]"\'])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def detect_domain_terms(text: str, lang: str = "kannada") -> List[Dict[str, str]]:
        """Identifies domain-specific terminology present in input string."""
        found = []
        words = text.split()
        if lang.lower() == "kannada":
            for word in words:
                # Strip punctuation for lookup
                clean_word = re.sub(r'[^\u0C80-\u0CFF]', '', word)
                if clean_word in AGRI_DOMAIN_LEXICON:
                    found.append({
                        "source_term": clean_word,
                        "domain": "Agriculture",
                        "target_meaning": AGRI_DOMAIN_LEXICON[clean_word]
                    })
        return found
