"""
Medical Text Summarization Module
=================================
Converts physician-patient transcripts into structured medical reports.
Uses LOCAL models only - no external API dependencies.

Methods:
1. Extractive summarization (rule-based, always available)
2. Abstractive summarization (T5/BART when available)
"""

import re
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

# Try to import transformers for local model
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Note: transformers not available. Using extractive summarization.")


@dataclass
class MedicalSummary:
    """Structured medical summary."""
    patient_name: str
    symptoms: List[str]
    diagnosis: str
    treatment: List[str]
    current_status: str
    prognosis: str
    additional_notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format (assignment output)."""
        return {
            "Patient_Name": self.patient_name,
            "Symptoms": self.symptoms,
            "Diagnosis": self.diagnosis,
            "Treatment": self.treatment,
            "Current_Status": self.current_status,
            "Prognosis": self.prognosis
        }


class MedicalSummarizer:
    """
    Summarize medical transcripts into structured reports.
    
    Features:
    - 100% LOCAL - no external API calls
    - Extractive summarization (rule-based)
    - Optional: HuggingFace T5 for abstractive summaries
    - NER integration for entity extraction
    """
    
    # Medical keywords for extraction
    SYMPTOM_KEYWORDS = [
        "pain", "ache", "discomfort", "hurt", "sore", "stiff", "swelling", 
        "fatigue", "tired", "weak", "nausea", "dizziness", "headache",
        "fever", "cough", "shortness of breath", "numbness", "tingling",
        "bleeding", "rash", "itching", "burning", "cramping"
    ]
    
    TREATMENT_KEYWORDS = [
        "physiotherapy", "physical therapy", "medication", "surgery",
        "painkillers", "antibiotics", "therapy", "treatment", "prescribed",
        "injection", "exercise", "rest", "ice", "compression",
        "rehabilitation", "counseling", "supplements"
    ]
    
    DIAGNOSIS_PATTERNS = [
        r"(?:diagnosed with|diagnosis is|diagnosis of|it's a?|it was a?|confirms?)\s+([A-Za-z\s]+(?:injury|disease|syndrome|disorder|condition)?)",
        r"(whiplash|fracture|sprain|strain|arthritis|diabetes|hypertension|asthma|GERD|migraine|vertigo)",
        r"you have\s+([A-Za-z\s]+)"
    ]
    
    def __init__(self, use_transformers: bool = True, model_path: Optional[str] = None):
        """
        Initialize the summarizer.
        
        Args:
            use_transformers: Whether to try loading transformer models
            model_path: Path to custom fine-tuned model (optional)
        """
        self.summarizer = None
        self.ner = None
        
        if use_transformers and TRANSFORMERS_AVAILABLE:
            self._load_models(model_path)
    
    def _load_models(self, model_path: Optional[str] = None):
        """Load local transformer models."""
        try:
            # Use a smaller, efficient summarization model
            model_name = model_path or "facebook/bart-large-cnn"
            
            # Try to load - will use CPU by default
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=-1,  # CPU
                torch_dtype=torch.float32
            )
            print(f"✓ Loaded summarization model: {model_name}")
            
        except Exception as e:
            print(f"Note: Could not load transformer model: {e}")
            print("Using extractive summarization (no dependencies required)")
            self.summarizer = None
    
    def summarize(self, transcript: str) -> MedicalSummary:
        """
        Summarize a medical transcript.
        
        Args:
            transcript: Physician-patient conversation text
            
        Returns:
            MedicalSummary object
        """
        # Always use extractive for structured extraction
        # Transformers can optionally provide an additional summary
        return self._extractive_summarize(transcript)
    
    def _extractive_summarize(self, transcript: str) -> MedicalSummary:
        """Rule-based extractive summarization - no dependencies needed."""
        
        # Extract patient name
        patient_name = self._extract_patient_name(transcript)
        
        # Extract symptoms using keyword matching
        symptoms = self._extract_symptoms(transcript)
        
        # Extract diagnosis
        diagnosis = self._extract_diagnosis(transcript)
        
        # Extract treatments
        treatments = self._extract_treatments(transcript)
        
        # Extract current status
        current_status = self._extract_current_status(transcript)
        
        # Extract prognosis
        prognosis = self._extract_prognosis(transcript)
        
        return MedicalSummary(
            patient_name=patient_name,
            symptoms=symptoms if symptoms else ["Not specified"],
            diagnosis=diagnosis,
            treatment=treatments if treatments else ["Not specified"],
            current_status=current_status,
            prognosis=prognosis
        )
    
    def _extract_patient_name(self, text: str) -> str:
        """Extract patient name from text."""
        patterns = [
            r"(?:Ms\.|Mrs\.|Mr\.|Miss|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?:patient|Patient)\s+([A-Z][a-z]+)",
            r"(?:Hello|Hi),?\s+(?:Ms\.|Mrs\.|Mr\.)?\s*([A-Z][a-z]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                # Try to get full name with title
                title_match = re.search(r"((?:Ms\.|Mrs\.|Mr\.)\s+[A-Z][a-z]+)", text)
                return title_match.group(1) if title_match else match.group(1)
        
        return "Unknown Patient"
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text."""
        symptoms = set()
        text_lower = text.lower()
        
        # Direct keyword matching
        for keyword in self.SYMPTOM_KEYWORDS:
            if keyword in text_lower:
                # Get surrounding context
                pattern = rf"(\w+\s+)?{keyword}(\s+\w+)?"
                matches = re.findall(pattern, text_lower)
                for match in matches:
                    symptom = ' '.join(filter(None, match)).strip()
                    if symptom:
                        symptoms.add(symptom)
                    else:
                        symptoms.add(keyword)
        
        # Pattern-based extraction
        symptom_patterns = [
            r"(?:having|experiencing|feel(?:ing)?|suffer(?:ing)? from)\s+([\w\s]+?)(?:\.|,|and|but|for)",
            r"(?:pain|ache|discomfort) in (?:my |the )?([\w\s]+?)(?:\.|,|and|but|for)",
            r"([\w\s]+?) (?:hurt|hurts|aches|is sore)",
        ]
        
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if len(match) < 50:  # Avoid capturing too much
                    symptoms.add(match.strip())
        
        # Clean up and remove duplicates
        cleaned = []
        for s in symptoms:
            s = s.strip()
            if s and len(s) > 2 and s not in ['i', 'my', 'the', 'a', 'an']:
                # Capitalize properly
                cleaned.append(s.title())
        
        return list(set(cleaned))[:10]  # Limit to 10 symptoms
    
    def _extract_diagnosis(self, text: str) -> str:
        """Extract diagnosis from text."""
        for pattern in self.DIAGNOSIS_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                diagnosis = match.group(1).strip()
                if len(diagnosis) > 3:
                    return diagnosis.title()
        
        return "Not specified"
    
    def _extract_treatments(self, text: str) -> List[str]:
        """Extract treatments from text."""
        treatments = set()
        text_lower = text.lower()
        
        # Keyword matching
        for keyword in self.TREATMENT_KEYWORDS:
            if keyword in text_lower:
                # Look for context
                pattern = rf"(\d+\s+(?:sessions?|weeks?|days?)?\s*(?:of\s+)?)?{keyword}"
                matches = re.findall(pattern, text_lower)
                for match in matches:
                    treatment = (match + " " + keyword).strip() if match else keyword
                    treatments.add(treatment.title())
        
        # Pattern-based extraction
        treatment_patterns = [
            r"(?:prescribed|taking|given|started on|recommend)\s+([\w\s]+\d*\s*(?:mg|ml)?)",
            r"(\d+\s+sessions?\s+of\s+[\w\s]+)",
            r"([\w]+\s*\d+\s*mg)",
        ]
        
        for pattern in treatment_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 3 and len(match) < 50:
                    treatments.add(match.strip())
        
        return list(treatments)[:10]
    
    def _extract_current_status(self, text: str) -> str:
        """Extract current status from text."""
        patterns = [
            r"(?:now|currently|still|at this point)\s+(?:I\s+)?(?:have|experiencing?|getting?|feeling?)\s+([\w\s]+?)(?:\.|,|but)",
            r"occasional ([\w\s]+?)(?:\.|,)",
            r"(?:better|worse|improved|improving)\s*,?\s*([\w\s]+?)(?:\.|,)",
            r"(?:doing|feeling)\s+(better|worse|good|fine|okay)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                status = match.group(1).strip()
                if status:
                    return status.capitalize()
        
        # Check for general improvement indicators
        if "better" in text.lower():
            return "Improving"
        elif "worse" in text.lower():
            return "Worsening"
        
        return "Not specified"
    
    def _extract_prognosis(self, text: str) -> str:
        """Extract prognosis from text."""
        patterns = [
            r"(?:expect|expected|expecting)\s+(?:you\s+)?(?:to\s+)?(?:make\s+)?(?:a\s+)?([\w\s]+(?:recovery|improvement)[\w\s]*)",
            r"(full recovery[\w\s]*)",
            r"(complete recovery[\w\s]*)",
            r"(?:should|will)\s+(recover|heal|improve)[\w\s]*",
            r"(no\s+long[\s-]term[\w\s]*)",
            r"prognosis is ([\w\s]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                prognosis = match.group(1).strip()
                if prognosis:
                    return prognosis.capitalize()
        
        return "Not specified"
    
    def get_abstractive_summary(self, transcript: str, max_length: int = 150) -> str:
        """
        Get an abstractive summary using transformer model.
        Only works if transformers are available.
        
        Args:
            transcript: Input text
            max_length: Maximum summary length
            
        Returns:
            Summary string
        """
        if self.summarizer is None:
            return "Transformer model not available."
        
        try:
            # Clean input
            clean_text = re.sub(r'\s+', ' ', transcript).strip()
            
            # Generate summary
            result = self.summarizer(
                clean_text,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )
            
            return result[0]['summary_text']
            
        except Exception as e:
            return f"Could not generate summary: {e}"
    
    def to_json(self, transcript: str) -> str:
        """Return summarization as JSON string."""
        summary = self.summarize(transcript)
        return json.dumps(summary.to_dict(), indent=2)


# Convenience function
def summarize_transcript(transcript: str) -> Dict:
    """
    Quick function to summarize a medical transcript.
    100% local - no API keys needed.
    
    Args:
        transcript: Input conversation
        
    Returns:
        Dictionary with structured summary
    """
    summarizer = MedicalSummarizer(use_transformers=False)  # Use rule-based for speed
    summary = summarizer.summarize(transcript)
    return summary.to_dict()


if __name__ == "__main__":
    sample = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    
    Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
    
    Physician: I see. Can you tell me more about the car accident and the symptoms you experienced afterward?
    
    Patient: I had pain in my neck and back almost right away. They said it was a whiplash injury.
    I had to take painkillers regularly and go through ten sessions of physiotherapy.
    
    Physician: I'm pleased to see your recovery. I'd expect you to make a full recovery within six months.
    """
    
    print("Testing LOCAL Medical Summarizer (No API Required)\n")
    print("=" * 50)
    
    result = summarize_transcript(sample)
    print(json.dumps(result, indent=2))
