"""
Medical Named Entity Recognition (NER) Module

This module extracts medical entities from physician-patient conversations:
- Symptoms
- Treatments
- Diagnoses
- Prognosis

Uses spaCy with custom entity patterns for medical domain.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import spacy
    from spacy.matcher import PhraseMatcher
    from spacy.tokens import Span
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not installed. Using regex-based extraction.")


@dataclass
class MedicalEntity:
    """Represents a medical entity extracted from text."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    context: str = ""


@dataclass
class NERResult:
    """Container for all extracted entities."""
    symptoms: List[MedicalEntity] = field(default_factory=list)
    treatments: List[MedicalEntity] = field(default_factory=list)
    diagnoses: List[MedicalEntity] = field(default_factory=list)
    prognosis: List[MedicalEntity] = field(default_factory=list)
    body_parts: List[MedicalEntity] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "symptoms": [{"text": e.text, "confidence": e.confidence} for e in self.symptoms],
            "treatments": [{"text": e.text, "confidence": e.confidence} for e in self.treatments],
            "diagnoses": [{"text": e.text, "confidence": e.confidence} for e in self.diagnoses],
            "prognosis": [{"text": e.text, "confidence": e.confidence} for e in self.prognosis],
            "body_parts": [{"text": e.text, "confidence": e.confidence} for e in self.body_parts]
        }
    
    def to_simple_dict(self) -> Dict:
        """Convert to simple list format (for structured output)."""
        return {
            "Symptoms": list(set(e.text.title() for e in self.symptoms)),
            "Treatment": list(set(e.text.title() for e in self.treatments)),
            "Diagnosis": self.diagnoses[0].text.title() if self.diagnoses else "Unknown",
            "Prognosis": self.prognosis[0].text if self.prognosis else "Unknown"
        }


class MedicalNERExtractor:
    """
    Extract medical entities from text using spaCy + custom patterns.
    
    Features:
    - Custom medical entity patterns
    - Confidence scoring
    - Context extraction
    - Handles ambiguous/missing data
    """
    
    def __init__(self, entities_path: Optional[str] = None):
        """
        Initialize the NER extractor.
        
        Args:
            entities_path: Path to medical_entities.json
        """
        self.entities = self._load_entities(entities_path)
        self.nlp = self._load_spacy_model()
        self._setup_matchers()
    
    def _load_entities(self, entities_path: Optional[str]) -> Dict:
        """Load medical entities from JSON file."""
        if entities_path is None:
            # Default path relative to this file
            entities_path = Path(__file__).parent.parent / "data" / "medical_entities.json"
        
        try:
            with open(entities_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Entities file not found at {entities_path}")
            return self._get_default_entities()
    
    def _get_default_entities(self) -> Dict:
        """Return default entities if file not found."""
        return {
            "symptoms": ["pain", "headache", "nausea", "fatigue", "stiffness"],
            "treatments": ["physiotherapy", "medication", "surgery", "rest"],
            "diagnoses": ["whiplash", "strain", "fracture", "concussion"],
            "prognosis_indicators": ["full recovery", "improving", "chronic"],
            "body_parts": ["neck", "back", "head", "spine", "shoulder"]
        }
    
    def _load_spacy_model(self):
        """Load spaCy model - prefer custom trained model."""
        if not SPACY_AVAILABLE:
            return None
        
        # Try custom trained model first
        custom_model_paths = [
            Path(__file__).parent.parent / "models" / "medical_ner_model",
            Path("models/medical_ner_model"),
            Path("./medical_ner_model"),
        ]
        
        for model_path in custom_model_paths:
            if model_path.exists():
                try:
                    nlp = spacy.load(str(model_path))
                    print(f"✅ Loaded custom NER model from {model_path}")
                    self._using_custom_model = True
                    return nlp
                except Exception as e:
                    print(f"Note: Could not load custom model: {e}")
        
        # Fall back to pre-trained models
        self._using_custom_model = False
        try:
            nlp = spacy.load("en_core_web_lg")
        except OSError:
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: No spaCy model found. Using blank model.")
                nlp = spacy.blank("en")
        
        return nlp
    
    def _setup_matchers(self):
        """Setup phrase matchers for each entity type."""
        if self.nlp is None:
            self.matchers = {}
            return
        
        self.matchers = {}
        
        # Create matchers for each entity type
        entity_types = {
            "SYMPTOM": self.entities.get("symptoms", []),
            "TREATMENT": self.entities.get("treatments", []),
            "DIAGNOSIS": self.entities.get("diagnoses", []),
            "PROGNOSIS": self.entities.get("prognosis_indicators", []),
            "BODY_PART": self.entities.get("body_parts", [])
        }
        
        for entity_type, phrases in entity_types.items():
            matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            patterns = [self.nlp.make_doc(phrase) for phrase in phrases]
            matcher.add(entity_type, patterns)
            self.matchers[entity_type] = matcher
    
    def extract(self, text: str) -> NERResult:
        """
        Extract all medical entities from text.
        
        Args:
            text: Input text (physician-patient conversation)
            
        Returns:
            NERResult with all extracted entities
        """
        result = NERResult()
        
        if self.nlp is None:
            # Fallback to regex extraction
            return self._regex_extract(text)
        
        doc = self.nlp(text)
        
        # If using custom trained model, use its NER directly
        if getattr(self, '_using_custom_model', False) and doc.ents:
            for ent in doc.ents:
                entity = MedicalEntity(
                    text=ent.text.lower(),
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.85,  # Custom model confidence
                    context=self._get_context_from_ent(ent, doc)
                )
                
                # Add to appropriate list based on label
                if ent.label_ == "SYMPTOM":
                    result.symptoms.append(entity)
                elif ent.label_ == "TREATMENT":
                    result.treatments.append(entity)
                elif ent.label_ == "DIAGNOSIS":
                    result.diagnoses.append(entity)
                elif ent.label_ == "DURATION":
                    # Duration can inform prognosis
                    result.prognosis.append(entity)
                elif ent.label_ == "BODY_PART":
                    result.body_parts.append(entity)
        else:
            # Fall back to phrase matchers
            for entity_type, matcher in self.matchers.items():
                matches = matcher(doc)
                
                for match_id, start, end in matches:
                    span = doc[start:end]
                    entity = MedicalEntity(
                        text=span.text.lower(),
                        label=entity_type,
                        start=span.start_char,
                        end=span.end_char,
                        confidence=self._calculate_confidence(span, doc),
                        context=self._get_context(span, doc)
                    )
                    
                    # Add to appropriate list
                    if entity_type == "SYMPTOM":
                        result.symptoms.append(entity)
                    elif entity_type == "TREATMENT":
                        result.treatments.append(entity)
                    elif entity_type == "DIAGNOSIS":
                        result.diagnoses.append(entity)
                    elif entity_type == "PROGNOSIS":
                        result.prognosis.append(entity)
                    elif entity_type == "BODY_PART":
                        result.body_parts.append(entity)
        
        # Deduplicate entities
        result = self._deduplicate(result)
        
        return result
    
    def _get_context_from_ent(self, ent, doc, window: int = 50) -> str:
        """Get surrounding context for an entity from spaCy entity."""
        start = max(0, ent.start_char - window)
        end = min(len(doc.text), ent.end_char + window)
        return doc.text[start:end]
    
    def _regex_extract(self, text: str) -> NERResult:
        """Fallback regex-based extraction when spaCy is not available."""
        result = NERResult()
        text_lower = text.lower()
        
        for symptom in self.entities.get("symptoms", []):
            if symptom in text_lower:
                result.symptoms.append(MedicalEntity(
                    text=symptom, label="SYMPTOM",
                    start=text_lower.find(symptom),
                    end=text_lower.find(symptom) + len(symptom),
                    confidence=0.8
                ))
        
        for treatment in self.entities.get("treatments", []):
            if treatment in text_lower:
                result.treatments.append(MedicalEntity(
                    text=treatment, label="TREATMENT",
                    start=text_lower.find(treatment),
                    end=text_lower.find(treatment) + len(treatment),
                    confidence=0.8
                ))
        
        for diagnosis in self.entities.get("diagnoses", []):
            if diagnosis in text_lower:
                result.diagnoses.append(MedicalEntity(
                    text=diagnosis, label="DIAGNOSIS",
                    start=text_lower.find(diagnosis),
                    end=text_lower.find(diagnosis) + len(diagnosis),
                    confidence=0.8
                ))
        
        for prognosis in self.entities.get("prognosis_indicators", []):
            if prognosis in text_lower:
                result.prognosis.append(MedicalEntity(
                    text=prognosis, label="PROGNOSIS",
                    start=text_lower.find(prognosis),
                    end=text_lower.find(prognosis) + len(prognosis),
                    confidence=0.8
                ))
        
        return self._deduplicate(result)
    
    def _calculate_confidence(self, span, doc) -> float:
        """
        Calculate confidence score for an entity.
        
        Factors:
        - Exact match vs partial match
        - Context (mentioned by doctor vs patient)
        - Frequency of mention
        """
        confidence = 0.9  # Base confidence for exact match
        
        # Boost if mentioned by physician
        sent = span.sent.text.lower()
        if "physician:" in sent or "doctor:" in sent:
            confidence += 0.05
        
        # Boost if mentioned multiple times
        text_lower = doc.text.lower()
        count = text_lower.count(span.text.lower())
        if count > 1:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _get_context(self, span, doc, window: int = 50) -> str:
        """Get surrounding context for an entity."""
        start = max(0, span.start_char - window)
        end = min(len(doc.text), span.end_char + window)
        return doc.text[start:end]
    
    def _deduplicate(self, result: NERResult) -> NERResult:
        """Remove duplicate entities, keeping highest confidence."""
        def dedupe_list(entities: List[MedicalEntity]) -> List[MedicalEntity]:
            seen = {}
            for entity in entities:
                key = entity.text.lower()
                if key not in seen or entity.confidence > seen[key].confidence:
                    seen[key] = entity
            return list(seen.values())
        
        result.symptoms = dedupe_list(result.symptoms)
        result.treatments = dedupe_list(result.treatments)
        result.diagnoses = dedupe_list(result.diagnoses)
        result.prognosis = dedupe_list(result.prognosis)
        result.body_parts = dedupe_list(result.body_parts)
        
        return result
    
    def extract_patient_name(self, text: str) -> Optional[str]:
        """Extract patient name from conversation."""
        # Look for patterns like "Ms. Jones", "Mr. Smith", etc.
        patterns = [
            r"(?:Ms\.|Mrs\.|Mr\.|Miss)\s+([A-Z][a-z]+)",
            r"(?:patient|Patient)\s+([A-Z][a-z]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return None
    
    def handle_ambiguous_data(self, result: NERResult, text: str) -> NERResult:
        """
        Handle ambiguous or missing medical data.
        
        Strategy:
        1. Flag low-confidence extractions
        2. Infer missing data from context
        3. Provide explanation for uncertainties
        """
        # If no diagnosis found, look for implicit mentions
        if not result.diagnoses:
            implicit_patterns = [
                (r"(?:diagnosed with|diagnosis of|it's a|it was)\s+(\w+(?:\s+\w+)?)", 0.7),
                (r"(?:suffering from|has|have)\s+(\w+(?:\s+\w+)?)\s+(?:injury|condition)", 0.6),
            ]
            
            for pattern, confidence in implicit_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    result.diagnoses.append(MedicalEntity(
                        text=match.group(1),
                        label="DIAGNOSIS",
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        context="Inferred from context"
                    ))
                    break
        
        return result


# Convenience function for quick extraction
def extract_medical_entities(text: str, entities_path: Optional[str] = None) -> Dict:
    """
    Quick function to extract medical entities from text.
    
    Args:
        text: Input text
        entities_path: Optional path to entities JSON
        
    Returns:
        Dictionary with extracted entities
    """
    extractor = MedicalNERExtractor(entities_path)
    result = extractor.extract(text)
    result = extractor.handle_ambiguous_data(result, text)
    return result.to_dict()


if __name__ == "__main__":
    # Test with sample text
    sample = """
    Doctor: How are you feeling today?
    Patient: I had a car accident. My neck and back hurt a lot for four weeks.
    Doctor: Did you receive treatment?
    Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
    """
    
    result = extract_medical_entities(sample)
    print(json.dumps(result, indent=2))
