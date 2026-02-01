"""
Sentiment and Intent Analysis Module

Analyzes patient sentiment and intent from medical conversations:
- Sentiment: Anxious, Neutral, Reassured
- Intent: Seeking reassurance, Reporting symptoms, Expressing concern, etc.

Uses transformer models for classification.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Using rule-based analysis.")


class Sentiment(Enum):
    ANXIOUS = "Anxious"
    NEUTRAL = "Neutral"
    REASSURED = "Reassured"


class Intent(Enum):
    SEEKING_REASSURANCE = "Seeking reassurance"
    REPORTING_SYMPTOMS = "Reporting symptoms"
    EXPRESSING_CONCERN = "Expressing concern"
    ASKING_QUESTIONS = "Asking questions"
    EXPRESSING_RELIEF = "Expressing relief"
    PROVIDING_HISTORY = "Providing medical history"
    CONFIRMING_UNDERSTANDING = "Confirming understanding"


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    sentiment: str
    confidence: float
    explanation: str
    
    def to_dict(self) -> Dict:
        return {
            "Sentiment": self.sentiment,
            "Confidence": round(self.confidence, 2),
            "Explanation": self.explanation
        }


@dataclass
class IntentResult:
    """Result of intent detection."""
    primary_intent: str
    secondary_intents: List[str]
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            "Intent": self.primary_intent,
            "Secondary_Intents": self.secondary_intents,
            "Confidence": round(self.confidence, 2)
        }


@dataclass
class AnalysisResult:
    """Combined sentiment and intent analysis result."""
    sentiment: SentimentResult
    intent: IntentResult
    sentiment_journey: List[Dict]  # Sentiment over conversation phases
    
    def to_dict(self) -> Dict:
        return {
            "Sentiment": self.sentiment.sentiment,
            "Intent": self.intent.primary_intent,
            "Confidence": round((self.sentiment.confidence + self.intent.confidence) / 2, 2),
            "Details": {
                "sentiment_analysis": self.sentiment.to_dict(),
                "intent_analysis": self.intent.to_dict(),
                "sentiment_journey": self.sentiment_journey
            }
        }
    
    def to_simple_dict(self) -> Dict:
        """Simple output format for structured output."""
        return {
            "Sentiment": self.sentiment.sentiment,
            "Intent": self.intent.primary_intent
        }


class MedicalSentimentAnalyzer:
    """
    Analyze patient sentiment and intent from medical conversations.
    
    Uses a combination of:
    1. Transformer-based sentiment classification
    2. Rule-based intent detection with medical context
    3. Keyword matching for medical-specific sentiment
    """
    
    # Keywords for rule-based analysis
    ANXIOUS_KEYWORDS = [
        "worried", "worry", "scared", "afraid", "anxious", "nervous",
        "concerning", "concerning", "fear", "terrified", "panic",
        "what if", "might be", "could it be", "scary", "frightened",
        "distressing", "overwhelming", "stressed", "trouble sleeping"
    ]
    
    REASSURED_KEYWORDS = [
        "relief", "relieved", "better", "great", "good to hear",
        "thank you", "appreciate", "encouraging", "positive",
        "glad", "happy", "confident", "optimistic", "improving",
        "that's a relief", "feel better", "good news"
    ]
    
    NEUTRAL_INDICATORS = [
        "okay", "fine", "alright", "yes", "no", "understand",
        "I see", "makes sense", "got it"
    ]
    
    # Intent patterns
    INTENT_PATTERNS = {
        Intent.SEEKING_REASSURANCE: [
            r"will I be okay",
            r"is it serious",
            r"should I worry",
            r"hope it gets better",
            r"don't need to worry",
            r"affecting me in the future",
            r"are you sure"
        ],
        Intent.REPORTING_SYMPTOMS: [
            r"I have",
            r"I feel",
            r"experiencing",
            r"my .* hurts?",
            r"pain in",
            r"trouble with",
            r"difficulty"
        ],
        Intent.EXPRESSING_CONCERN: [
            r"worried about",
            r"concerned about",
            r"afraid of",
            r"scared that",
            r"what if",
            r"might be"
        ],
        Intent.ASKING_QUESTIONS: [
            r"what should I",
            r"when do I",
            r"how long",
            r"will I need",
            r"do I need to",
            r"\?"
        ],
        Intent.EXPRESSING_RELIEF: [
            r"that's a relief",
            r"glad to hear",
            r"thank you",
            r"appreciate",
            r"feel better",
            r"great to hear"
        ],
        Intent.PROVIDING_HISTORY: [
            r"I had",
            r"I was",
            r"happened on",
            r"started when",
            r"since then",
            r"at that time"
        ]
    }
    
    def __init__(self, use_transformer: bool = True, model_path: str = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            use_transformer: Whether to use transformer model (if available)
            model_path: Path to custom trained model (optional)
        """
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self.sentiment_pipeline = None
        self.custom_model = None
        self.custom_tokenizer = None
        self.id2label = {0: "Anxious", 1: "Neutral", 2: "Reassured"}
        
        # Try to load custom trained model first
        if self.use_transformer:
            self._load_custom_model(model_path)
            if self.custom_model is None:
                self._load_transformer_model()
    
    def _load_custom_model(self, model_path: str = None):
        """Load custom-trained medical sentiment model."""
        import os
        from pathlib import Path
        
        # Default path to trained model
        if model_path is None:
            # Try multiple locations
            possible_paths = [
                Path(__file__).parent.parent / "models" / "medical_sentiment_production",
                Path("models/medical_sentiment_production"),
                Path("./medical_sentiment_production"),
            ]
            for p in possible_paths:
                if p.exists():
                    model_path = str(p)
                    break
        
        if model_path and os.path.exists(model_path):
            try:
                self.custom_tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.custom_model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.custom_model.eval()
                print(f"✅ Loaded custom medical sentiment model from {model_path}")
            except Exception as e:
                print(f"Note: Could not load custom model: {e}")
                self.custom_model = None
    
    def _load_transformer_model(self):
        """Load pre-trained sentiment analysis model (fallback)."""
        try:
            # Use DistilBERT fine-tuned on SST-2
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # CPU
            )
        except Exception as e:
            print(f"Warning: Could not load transformer model: {e}")
            self.use_transformer = False
    
    def analyze(self, text: str) -> AnalysisResult:
        """
        Perform complete sentiment and intent analysis.
        
        Args:
            text: Patient's dialogue or full conversation
            
        Returns:
            AnalysisResult with sentiment, intent, and journey
        """
        # Extract patient dialogues only
        patient_text = self._extract_patient_dialogue(text)
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(patient_text)
        
        # Detect intent
        intent = self._detect_intent(patient_text)
        
        # Calculate sentiment journey
        journey = self._calculate_sentiment_journey(text)
        
        return AnalysisResult(
            sentiment=sentiment,
            intent=intent,
            sentiment_journey=journey
        )
    
    def _extract_patient_dialogue(self, text: str) -> str:
        """Extract only patient's dialogue from conversation."""
        patient_lines = []
        
        for line in text.split('\n'):
            line = line.strip()
            if line.lower().startswith('patient:'):
                # Remove the "Patient:" prefix
                content = re.sub(r'^patient:\s*', '', line, flags=re.IGNORECASE)
                patient_lines.append(content)
        
        return ' '.join(patient_lines) if patient_lines else text
    
    def _analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of text."""
        # Use custom trained model if available
        if self.custom_model is not None:
            return self._custom_model_sentiment(text)
        elif self.use_transformer and self.sentiment_pipeline:
            return self._transformer_sentiment(text)
        else:
            return self._rule_based_sentiment(text)
    
    def _custom_model_sentiment(self, text: str) -> SentimentResult:
        """Use custom-trained medical sentiment model."""
        try:
            # Tokenize
            inputs = self.custom_tokenizer(
                text[:512], 
                return_tensors="pt", 
                truncation=True, 
                max_length=128
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.custom_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_id = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_id].item()
            
            sentiment = self.id2label[pred_id]
            
            return SentimentResult(
                sentiment=sentiment,
                confidence=confidence,
                explanation=f"Custom medical model prediction ({confidence:.1%} confidence)"
            )
            
        except Exception as e:
            print(f"Custom model failed: {e}")
            return self._rule_based_sentiment(text)
    
    def _transformer_sentiment(self, text: str) -> SentimentResult:
        """Use generic transformer model for sentiment analysis (fallback)."""
        try:
            # Get base sentiment from transformer
            result = self.sentiment_pipeline(text[:512])[0]  # Limit length
            
            # Map to our medical sentiment categories
            base_label = result['label']
            base_score = result['score']
            
            # Adjust based on medical keywords
            anxious_score = self._count_keywords(text, self.ANXIOUS_KEYWORDS)
            reassured_score = self._count_keywords(text, self.REASSURED_KEYWORDS)
            
            # Determine final sentiment
            if anxious_score > reassured_score:
                sentiment = Sentiment.ANXIOUS.value
                confidence = min(0.9, base_score + (anxious_score * 0.1))
                explanation = f"Detected {anxious_score} anxious indicators"
            elif reassured_score > anxious_score:
                sentiment = Sentiment.REASSURED.value
                confidence = min(0.9, base_score + (reassured_score * 0.1))
                explanation = f"Detected {reassured_score} positive indicators"
            else:
                sentiment = Sentiment.NEUTRAL.value
                confidence = base_score * 0.8
                explanation = "No strong emotional indicators detected"
            
            return SentimentResult(
                sentiment=sentiment,
                confidence=confidence,
                explanation=explanation
            )
            
        except Exception as e:
            print(f"Transformer analysis failed: {e}")
            return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> SentimentResult:
        """Fallback rule-based sentiment analysis."""
        text_lower = text.lower()
        
        anxious_score = self._count_keywords(text_lower, self.ANXIOUS_KEYWORDS)
        reassured_score = self._count_keywords(text_lower, self.REASSURED_KEYWORDS)
        
        if anxious_score > reassured_score and anxious_score > 0:
            return SentimentResult(
                sentiment=Sentiment.ANXIOUS.value,
                confidence=min(0.85, 0.5 + (anxious_score * 0.1)),
                explanation=f"Found {anxious_score} anxiety indicators: worry, scared, concerned"
            )
        elif reassured_score > anxious_score and reassured_score > 0:
            return SentimentResult(
                sentiment=Sentiment.REASSURED.value,
                confidence=min(0.85, 0.5 + (reassured_score * 0.1)),
                explanation=f"Found {reassured_score} positive indicators: relief, better, glad"
            )
        else:
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL.value,
                confidence=0.7,
                explanation="No strong emotional indicators detected"
            )
    
    def _count_keywords(self, text: str, keywords: List[str]) -> int:
        """Count occurrences of keywords in text."""
        text_lower = text.lower()
        count = 0
        for keyword in keywords:
            if keyword in text_lower:
                count += 1
        return count
    
    def _detect_intent(self, text: str) -> IntentResult:
        """Detect patient intent from text."""
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                score += len(matches)
            intent_scores[intent] = score
        
        # Sort by score
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get primary and secondary intents
        primary = sorted_intents[0] if sorted_intents else (Intent.REPORTING_SYMPTOMS, 0)
        secondary = [intent.value for intent, score in sorted_intents[1:4] if score > 0]
        
        # Calculate confidence based on match strength
        total_matches = sum(intent_scores.values())
        confidence = min(0.9, 0.5 + (primary[1] / max(total_matches, 1)) * 0.4)
        
        return IntentResult(
            primary_intent=primary[0].value,
            secondary_intents=secondary,
            confidence=confidence
        )
    
    def _calculate_sentiment_journey(self, text: str) -> List[Dict]:
        """
        Track sentiment changes throughout the conversation.
        
        Returns list of sentiment readings at different phases.
        """
        journey = []
        
        # Split conversation into phases
        lines = text.split('\n')
        phases = self._split_into_phases(lines)
        
        phase_names = ["Opening", "History", "Examination", "Assessment", "Closing"]
        
        for i, phase in enumerate(phases):
            if not phase.strip():
                continue
                
            patient_text = self._extract_patient_dialogue(phase)
            if patient_text:
                sentiment = self._rule_based_sentiment(patient_text)
                journey.append({
                    "phase": phase_names[min(i, len(phase_names) - 1)],
                    "sentiment": sentiment.sentiment,
                    "confidence": sentiment.confidence
                })
        
        return journey if journey else [{"phase": "Overall", "sentiment": "Neutral", "confidence": 0.7}]
    
    def _split_into_phases(self, lines: List[str], num_phases: int = 5) -> List[str]:
        """Split conversation lines into phases."""
        if len(lines) <= num_phases:
            return ['\n'.join(lines)]
        
        phase_size = len(lines) // num_phases
        phases = []
        
        for i in range(num_phases):
            start = i * phase_size
            end = start + phase_size if i < num_phases - 1 else len(lines)
            phases.append('\n'.join(lines[start:end]))
        
        return phases


# Convenience function
def analyze_sentiment(text: str) -> Dict:
    """
    Quick function to analyze sentiment and intent.
    
    Args:
        text: Input text (patient dialogue or conversation)
        
    Returns:
        Dictionary with sentiment and intent analysis
    """
    analyzer = MedicalSentimentAnalyzer()
    result = analyzer.analyze(text)
    return result.to_dict()


if __name__ == "__main__":
    # Test with sample text
    sample = "I'm a bit worried about my back pain, but I hope it gets better soon."
    
    result = analyze_sentiment(sample)
    print(f"Sentiment: {result['Sentiment']}")
    print(f"Intent: {result['Intent']}")
