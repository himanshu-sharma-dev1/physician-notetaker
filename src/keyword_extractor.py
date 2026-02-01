"""
Medical Keyword Extraction Module

Extracts important medical phrases and keywords from transcripts.
Uses TF-IDF and domain-specific vocabulary matching.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
from dataclasses import dataclass

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class KeywordResult:
    """Container for extracted keywords."""
    keywords: List[Tuple[str, float]]  # (keyword, score)
    medical_terms: List[str]
    phrases: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "keywords": [{"term": k, "score": round(s, 3)} for k, s in self.keywords[:10]],
            "medical_terms": self.medical_terms,
            "key_phrases": self.phrases
        }
    
    def get_top_keywords(self, n: int = 10) -> List[str]:
        """Get top N keywords."""
        return [k for k, s in self.keywords[:n]]


class MedicalKeywordExtractor:
    """
    Extract medical keywords and phrases from text.
    
    Methods:
    - TF-IDF based keyword importance
    - Custom medical vocabulary matching
    - N-gram extraction for multi-word phrases
    """
    
    # Common non-medical stopwords to filter
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "just",
        "but", "and", "or", "if", "because", "until", "while",
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him",
        "his", "himself", "she", "her", "hers", "herself", "it", "its",
        "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "yes", "no", "okay", "ok", "well", "like", "just", "really",
        "doctor", "patient", "physician", "good", "morning", "afternoon",
        "hello", "hi", "thank", "thanks", "please", "sorry", "today"
    }
    
    # Medical-specific keywords to prioritize
    MEDICAL_TERMS = {
        "pain", "injury", "treatment", "diagnosis", "symptoms", "recovery",
        "therapy", "medication", "surgery", "examination", "condition",
        "chronic", "acute", "mild", "severe", "moderate", "inflammation",
        "fracture", "strain", "sprain", "whiplash", "concussion",
        "physiotherapy", "physical therapy", "rehabilitation", "prognosis",
        "X-ray", "MRI", "CT scan", "blood test", "prescription",
        "painkillers", "anti-inflammatory", "muscle relaxants",
        "tenderness", "stiffness", "mobility", "range of motion",
        "follow-up", "appointment", "referral", "specialist"
    }
    
    def __init__(self, entities_path: Optional[str] = None):
        """Initialize keyword extractor."""
        self.medical_vocabulary = self._load_medical_vocabulary(entities_path)
    
    def _load_medical_vocabulary(self, entities_path: Optional[str]) -> Set[str]:
        """Load medical vocabulary from entities file."""
        vocab = set(self.MEDICAL_TERMS)
        
        if entities_path is None:
            entities_path = Path(__file__).parent.parent / "data" / "medical_entities.json"
        
        try:
            with open(entities_path, 'r') as f:
                entities = json.load(f)
                for key in ["symptoms", "treatments", "diagnoses", "prognosis_indicators", "body_parts"]:
                    if key in entities:
                        vocab.update(entities[key])
        except FileNotFoundError:
            pass
        
        return vocab
    
    def extract(self, text: str) -> KeywordResult:
        """
        Extract keywords and phrases from text.
        
        Args:
            text: Input text
            
        Returns:
            KeywordResult with keywords, medical terms, and phrases
        """
        # Clean text
        clean_text = self._preprocess(text)
        
        # Extract using different methods
        tfidf_keywords = self._tfidf_extract(clean_text)
        medical_terms = self._extract_medical_terms(text)
        phrases = self._extract_phrases(text)
        
        # Combine and rank
        combined_keywords = self._combine_keywords(tfidf_keywords, medical_terms)
        
        return KeywordResult(
            keywords=combined_keywords,
            medical_terms=medical_terms,
            phrases=phrases
        )
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text for keyword extraction."""
        # Remove speaker labels
        text = re.sub(r'(?:Physician|Doctor|Patient):\s*', '', text, flags=re.IGNORECASE)
        # Remove special characters (keep hyphens for medical terms)
        text = re.sub(r'[^\w\s\-]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()
    
    def _tfidf_extract(self, text: str) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF."""
        if not SKLEARN_AVAILABLE:
            return self._frequency_extract(text)
        
        try:
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words=list(self.STOPWORDS),
                max_features=50
            )
            
            # Fit on the text (as a single document)
            tfidf_matrix = vectorizer.fit_transform([text])
            
            # Get feature names and scores
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Sort by score
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Filter out low-value keywords
            return [(k, s) for k, s in keyword_scores if s > 0.05]
            
        except Exception as e:
            print(f"TF-IDF extraction failed: {e}")
            return self._frequency_extract(text)
    
    def _frequency_extract(self, text: str) -> List[Tuple[str, float]]:
        """Fallback frequency-based extraction."""
        words = text.split()
        # Filter stopwords
        words = [w for w in words if w not in self.STOPWORDS and len(w) > 2]
        
        # Count frequencies
        counts = Counter(words)
        total = sum(counts.values())
        
        # Normalize to scores
        return [(word, count / total) for word, count in counts.most_common(20)]
    
    def _extract_medical_terms(self, text: str) -> List[str]:
        """Extract known medical terms from text."""
        text_lower = text.lower()
        found_terms = []
        
        for term in self.medical_vocabulary:
            if term.lower() in text_lower:
                found_terms.append(term)
        
        # Deduplicate and return
        return list(set(found_terms))
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract important medical phrases using patterns."""
        phrases = []
        
        patterns = [
            # Treatment phrases
            r"(\d+\s+(?:sessions?|treatments?|weeks?|days?|months?)\s+of\s+\w+(?:\s+\w+)?)",
            r"((?:physiotherapy|physical therapy|occupational therapy)\s+sessions?)",
            
            # Symptom phrases
            r"((?:neck|back|head|chest|abdominal)\s+pain)",
            r"(trouble\s+\w+ing)",
            r"(difficulty\s+\w+ing)",
            
            # Diagnosis phrases
            r"(\w+\s+injury)",
            r"(\w+\s+strain)",
            r"(\w+\s+fracture)",
            
            # Status phrases
            r"(full\s+recovery)",
            r"(full range of (?:motion|movement))",
            r"(no\s+(?:lasting|long-term)\s+damage)",
            
            # Time references
            r"((?:first|last)\s+(?:few|several|four|six|eight|ten)\s+weeks?)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            phrases.extend(matches)
        
        # Clean and deduplicate
        phrases = [p.strip().lower() for p in phrases]
        return list(set(phrases))
    
    def _combine_keywords(
        self, 
        tfidf_keywords: List[Tuple[str, float]], 
        medical_terms: List[str]
    ) -> List[Tuple[str, float]]:
        """Combine and rank keywords from different sources."""
        keyword_scores = dict(tfidf_keywords)
        
        # Boost medical terms
        for term in medical_terms:
            term_lower = term.lower()
            if term_lower in keyword_scores:
                keyword_scores[term_lower] *= 1.5
            else:
                keyword_scores[term_lower] = 0.3
        
        # Sort by score
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_keywords


# Convenience function
def extract_keywords(text: str) -> Dict:
    """
    Quick function to extract keywords from text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with keywords, medical terms, and phrases
    """
    extractor = MedicalKeywordExtractor()
    result = extractor.extract(text)
    return result.to_dict()


if __name__ == "__main__":
    sample = """
    Doctor: How are you feeling today?
    Patient: I had a car accident. My neck and back hurt a lot for four weeks.
    Doctor: Did you receive treatment?
    Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
    """
    
    result = extract_keywords(sample)
    print(json.dumps(result, indent=2))
