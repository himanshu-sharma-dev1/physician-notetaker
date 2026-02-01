"""
Main Pipeline Orchestration Module

Combines all NLP modules into a unified pipeline for:
- Medical NER extraction
- Text summarization
- Sentiment and intent analysis
- Keyword extraction
- SOAP note generation
"""

import json
from typing import Dict, Optional
from dataclasses import dataclass

from .medical_ner import MedicalNERExtractor, NERResult
from .summarizer import MedicalSummarizer, MedicalSummary
from .sentiment_analyzer import MedicalSentimentAnalyzer, AnalysisResult
from .keyword_extractor import MedicalKeywordExtractor, KeywordResult
from .soap_generator import SOAPNoteGenerator, SOAPNote


@dataclass
class PipelineResult:
    """Complete pipeline analysis result."""
    summary: MedicalSummary
    entities: NERResult
    sentiment: AnalysisResult
    keywords: KeywordResult
    soap_note: SOAPNote
    
    def to_dict(self) -> Dict:
        """Convert all results to dictionary."""
        return {
            "medical_summary": self.summary.to_dict(),
            "entities": self.entities.to_dict(),
            "sentiment_analysis": self.sentiment.to_dict(),
            "keywords": self.keywords.to_dict(),
            "soap_note": self.soap_note.to_dict()
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class PhysicianNotetaker:
    """
    Main pipeline orchestrator for the Physician Notetaker system.
    
    Combines:
    - Medical NER extraction
    - Text summarization (Gemini-powered)
    - Sentiment and intent analysis
    - Keyword extraction
    - SOAP note generation
    
    Usage:
        pipeline = PhysicianNotetaker(api_key="your-gemini-key")
        result = pipeline.process(transcript)
        print(result.to_json())
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        entities_path: Optional[str] = None
    ):
        """
        Initialize the pipeline with all modules.
        
        Args:
            api_key: Gemini API key for summarization and SOAP
            entities_path: Path to medical entities JSON
        """
        self.ner_extractor = MedicalNERExtractor(entities_path)
        self.summarizer = MedicalSummarizer(api_key)
        self.sentiment_analyzer = MedicalSentimentAnalyzer()
        self.keyword_extractor = MedicalKeywordExtractor(entities_path)
        self.soap_generator = SOAPNoteGenerator(api_key)
    
    def process(self, transcript: str) -> PipelineResult:
        """
        Process a medical transcript through the full pipeline.
        
        Args:
            transcript: Physician-patient conversation text
            
        Returns:
            PipelineResult with all analysis components
        """
        # 1. Extract medical entities
        entities = self.ner_extractor.extract(transcript)
        entities = self.ner_extractor.handle_ambiguous_data(entities, transcript)
        
        # 2. Generate medical summary
        summary = self.summarizer.summarize(transcript)
        
        # 3. Analyze sentiment and intent
        sentiment = self.sentiment_analyzer.analyze(transcript)
        
        # 4. Extract keywords
        keywords = self.keyword_extractor.extract(transcript)
        
        # 5. Generate SOAP note
        soap_note = self.soap_generator.generate(transcript)
        
        return PipelineResult(
            summary=summary,
            entities=entities,
            sentiment=sentiment,
            keywords=keywords,
            soap_note=soap_note
        )
    
    def get_summary(self, transcript: str) -> Dict:
        """Get only the medical summary."""
        entities = self.ner_extractor.extract(transcript)
        summary = self.summarizer.summarize(transcript)
        
        # Merge NER results with summary
        result = summary.to_dict()
        if not result.get("Symptoms") or result["Symptoms"] == ["Not specified"]:
            result["Symptoms"] = [e.text.title() for e in entities.symptoms]
        
        return result
    
    def get_sentiment(self, transcript: str) -> Dict:
        """Get only sentiment and intent analysis."""
        return self.sentiment_analyzer.analyze(transcript).to_simple_dict()
    
    def get_soap_note(self, transcript: str) -> Dict:
        """Get only the SOAP note."""
        return self.soap_generator.generate(transcript).to_dict()
    
    def get_keywords(self, transcript: str) -> Dict:
        """Get only keywords and phrases."""
        return self.keyword_extractor.extract(transcript).to_dict()
    
    def export_results(
        self, 
        result: PipelineResult, 
        format: str = "json"
    ) -> str:
        """
        Export results in specified format.
        
        Args:
            result: PipelineResult to export
            format: Output format ("json" or "clinical")
            
        Returns:
            Formatted string
        """
        if format == "json":
            return result.to_json()
        elif format == "clinical":
            return result.soap_note.to_clinical_format()
        else:
            raise ValueError(f"Unknown format: {format}")


# CLI interface
def main():
    """Command-line interface for the pipeline."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Physician Notetaker - Medical Transcript Analysis"
    )
    parser.add_argument(
        "--input", "-i", 
        type=str, 
        help="Input file path or use stdin"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        choices=["json", "clinical"],
        default="json",
        help="Output format"
    )
    parser.add_argument(
        "--component", "-c",
        type=str,
        choices=["all", "summary", "sentiment", "soap", "keywords"],
        default="all",
        help="Component to run"
    )
    
    args = parser.parse_args()
    
    # Read input
    if args.input:
        with open(args.input, 'r') as f:
            transcript = f.read()
    else:
        transcript = sys.stdin.read()
    
    # Process
    pipeline = PhysicianNotetaker()
    
    if args.component == "all":
        result = pipeline.process(transcript)
        print(pipeline.export_results(result, args.output))
    elif args.component == "summary":
        print(json.dumps(pipeline.get_summary(transcript), indent=2))
    elif args.component == "sentiment":
        print(json.dumps(pipeline.get_sentiment(transcript), indent=2))
    elif args.component == "soap":
        print(json.dumps(pipeline.get_soap_note(transcript), indent=2))
    elif args.component == "keywords":
        print(json.dumps(pipeline.get_keywords(transcript), indent=2))


if __name__ == "__main__":
    main()
