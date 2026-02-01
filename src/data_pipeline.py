"""
ETL Data Pipeline for Medical NLP
=================================
Extract, Transform, Load pipeline for processing medical conversation data.

This module handles:
- Data ingestion from multiple sources (JSON, TXT, CSV)
- Data cleaning and preprocessing
- Text normalization and tokenization
- Data validation and quality checks
- Train/validation/test splitting
- Feature extraction for ML models

Author: Himanshu Sharma
Author: Himanshu Sharma
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataStats:
    """Statistics about the processed dataset."""
    total_samples: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    avg_text_length: float = 0.0
    min_text_length: int = 0
    max_text_length: int = 0
    missing_values: int = 0
    duplicates_removed: int = 0
    invalid_samples: int = 0


@dataclass
class ProcessedData:
    """Container for processed dataset."""
    train: List[Dict]
    validation: List[Dict]
    test: List[Dict]
    stats: DataStats


class MedicalDataPipeline:
    """
    ETL Pipeline for Medical Conversation Data.
    
    Handles the complete data processing workflow:
    1. Extract: Load data from various sources
    2. Transform: Clean, normalize, and validate
    3. Load: Split and prepare for ML training
    """
    
    # Text cleaning patterns
    NOISE_PATTERNS = [
        r'\[.*?\]',           # Remove bracketed content like [inaudible]
        r'\(.*?\)',           # Remove parenthetical notes
        r'<.*?>',             # Remove HTML-like tags
        r'\.{3,}',            # Replace multiple dots
        r'-{2,}',             # Replace multiple dashes
        r'\s+',               # Normalize whitespace
    ]
    
    # Medical text normalization
    CONTRACTIONS = {
        "don't": "do not",
        "doesn't": "does not",
        "can't": "cannot",
        "won't": "will not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "I'm": "I am",
        "I've": "I have",
        "I'll": "I will",
        "I'd": "I would",
        "you're": "you are",
        "you've": "you have",
        "you'll": "you will",
        "you'd": "you would",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "we're": "we are",
        "they're": "they are",
        "that's": "that is",
        "what's": "what is",
        "there's": "there is",
    }
    
    # Valid sentiment labels
    VALID_SENTIMENTS = {"anxious", "neutral", "reassured"}
    
    # Valid intent labels
    VALID_INTENTS = {
        "reporting_symptoms",
        "expressing_concern", 
        "seeking_reassurance",
        "confirming_understanding",
        "asking_questions",
        "expressing_relief"
    }
    
    def __init__(self, data_dir: str = "data/training"):
        """
        Initialize the ETL pipeline.
        
        Args:
            data_dir: Directory containing training data files
        """
        self.data_dir = Path(data_dir)
        self.stats = DataStats()
        self.random_seed = 42
        random.seed(self.random_seed)
        
    # ==================== EXTRACT ====================
    
    def extract_json(self, filepath: str) -> List[Dict]:
        """
        Extract data from a JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of data records
        """
        full_path = self.data_dir / filepath
        logger.info(f"📥 Extracting data from: {full_path}")
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"   ✓ Loaded {len(data)} records")
            return data
        except FileNotFoundError:
            logger.error(f"   ✗ File not found: {full_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"   ✗ JSON decode error: {e}")
            return []
    
    def extract_txt(self, filepath: str) -> str:
        """
        Extract text from a plain text file.
        
        Args:
            filepath: Path to text file
            
        Returns:
            File contents as string
        """
        full_path = self.data_dir / filepath
        logger.info(f"📥 Extracting text from: {full_path}")
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"   ✓ Loaded {len(content)} characters")
            return content
        except FileNotFoundError:
            logger.error(f"   ✗ File not found: {full_path}")
            return ""
    
    def extract_all(self) -> Dict[str, Any]:
        """
        Extract data from all available sources.
        
        Returns:
            Dictionary containing all extracted data
        """
        logger.info("=" * 50)
        logger.info("🔄 EXTRACT PHASE")
        logger.info("=" * 50)
        
        extracted = {}
        
        # Load sentiment dataset
        sentiment_data = self.extract_json("sentiment_dataset_large.json")
        if sentiment_data:
            extracted['sentiment'] = sentiment_data
        
        # Load NER training data
        ner_data = self.extract_json("ner_training_data.json")
        if ner_data:
            extracted['ner'] = ner_data
        
        # Load intent data
        intent_data = self.extract_json("intent_training_data.json")
        if intent_data:
            extracted['intent'] = intent_data
        
        # Load full conversation data
        conversation_data = self.extract_json("medical_conversations.json")
        if conversation_data:
            extracted['conversations'] = conversation_data
        
        logger.info(f"\n📊 Extracted {len(extracted)} data sources")
        return extracted
    
    # ==================== TRANSFORM ====================
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase for processing
        cleaned = text.strip()
        
        # Expand contractions
        for contraction, expansion in self.CONTRACTIONS.items():
            cleaned = re.sub(
                re.escape(contraction), 
                expansion, 
                cleaned, 
                flags=re.IGNORECASE
            )
        
        # Remove noise patterns
        cleaned = re.sub(r'\[.*?\]', '', cleaned)  # [inaudible], [pause]
        cleaned = re.sub(r'<.*?>', '', cleaned)     # HTML tags
        cleaned = re.sub(r'\.{3,}', '...', cleaned) # Normalize ellipsis
        cleaned = re.sub(r'-{2,}', '-', cleaned)    # Normalize dashes
        cleaned = re.sub(r'\s+', ' ', cleaned)      # Normalize whitespace
        
        return cleaned.strip()
    
    def validate_sentiment_record(self, record: Dict) -> bool:
        """
        Validate a sentiment training record.
        
        Args:
            record: Data record to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if 'text' not in record or 'sentiment' not in record:
            return False
        
        # Check text is non-empty
        if not record['text'] or len(record['text'].strip()) < 5:
            return False
        
        # Check valid sentiment label
        if record['sentiment'] not in self.VALID_SENTIMENTS:
            return False
        
        # Check text length (reasonable bounds)
        if len(record['text']) > 1000:
            return False
        
        return True
    
    def validate_ner_record(self, record: List) -> bool:
        """
        Validate a NER training record.
        
        Args:
            record: NER record [text, entities_list]
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(record, list) or len(record) != 2:
            return False
        
        text, entities = record
        
        # Check text
        if not text or len(text.strip()) < 5:
            return False
        
        # Check entities structure
        if not isinstance(entities, list):
            return False
        
        # Validate each entity span
        for entity in entities:
            if not isinstance(entity, dict):
                return False
            if 'start' not in entity or 'end' not in entity or 'label' not in entity:
                return False
            if entity['start'] >= entity['end']:
                return False
            if entity['end'] > len(text):
                return False
        
        return True
    
    def remove_duplicates(self, data: List[Dict], key: str = 'text') -> List[Dict]:
        """
        Remove duplicate records based on a key field.
        
        Args:
            data: List of records
            key: Field to check for duplicates
            
        Returns:
            Deduplicated list
        """
        seen = set()
        unique = []
        duplicates = 0
        
        for record in data:
            text = record.get(key, '').lower().strip()
            if text not in seen:
                seen.add(text)
                unique.append(record)
            else:
                duplicates += 1
        
        self.stats.duplicates_removed += duplicates
        logger.info(f"   ✓ Removed {duplicates} duplicates")
        
        return unique
    
    def balance_classes(
        self, 
        data: List[Dict], 
        label_key: str = 'sentiment',
        strategy: str = 'undersample'
    ) -> List[Dict]:
        """
        Balance class distribution in the dataset.
        
        Args:
            data: List of records
            label_key: Key containing the label
            strategy: 'undersample' or 'oversample'
            
        Returns:
            Balanced dataset
        """
        # Group by class
        by_class = {}
        for record in data:
            label = record.get(label_key)
            if label not in by_class:
                by_class[label] = []
            by_class[label].append(record)
        
        # Get class counts
        counts = {k: len(v) for k, v in by_class.items()}
        logger.info(f"   Class distribution before balancing: {counts}")
        
        if strategy == 'undersample':
            # Undersample to minority class
            min_count = min(counts.values())
            balanced = []
            for label, records in by_class.items():
                random.shuffle(records)
                balanced.extend(records[:min_count])
        else:
            # Oversample to majority class
            max_count = max(counts.values())
            balanced = []
            for label, records in by_class.items():
                while len(records) < max_count:
                    records.extend(records[:max_count - len(records)])
                balanced.extend(records[:max_count])
        
        random.shuffle(balanced)
        
        # Log new distribution
        new_counts = Counter(r[label_key] for r in balanced)
        logger.info(f"   Class distribution after balancing: {dict(new_counts)}")
        
        return balanced
    
    def transform_sentiment_data(self, data: List[Dict]) -> List[Dict]:
        """
        Transform sentiment training data.
        
        Args:
            data: Raw sentiment data
            
        Returns:
            Transformed data
        """
        logger.info("\n📝 Transforming sentiment data...")
        
        transformed = []
        invalid = 0
        
        for record in data:
            # Validate
            if not self.validate_sentiment_record(record):
                invalid += 1
                continue
            
            # Clean text
            cleaned_text = self.clean_text(record['text'])
            
            # Create transformed record
            transformed.append({
                'text': cleaned_text,
                'sentiment': record['sentiment'],
                'intent': record.get('intent', 'unknown'),
                'text_length': len(cleaned_text.split())
            })
        
        self.stats.invalid_samples += invalid
        
        # Remove duplicates
        transformed = self.remove_duplicates(transformed)
        
        # Balance classes
        transformed = self.balance_classes(transformed, 'sentiment')
        
        logger.info(f"   ✓ Valid samples: {len(transformed)}")
        logger.info(f"   ✗ Invalid samples: {invalid}")
        
        return transformed
    
    def transform_ner_data(self, data: List) -> List[Tuple[str, Dict]]:
        """
        Transform NER training data to spaCy format.
        
        Args:
            data: Raw NER data
            
        Returns:
            Transformed data in spaCy format
        """
        logger.info("\n📝 Transforming NER data...")
        
        transformed = []
        invalid = 0
        
        for record in data:
            if not self.validate_ner_record(record):
                invalid += 1
                continue
            
            text, entities = record
            
            # Clean text (but preserve entity positions!)
            # For NER, we clean minimally to preserve spans
            cleaned_text = text.strip()
            
            # Convert entities to spaCy format
            spacy_entities = []
            for ent in entities:
                spacy_entities.append((ent['start'], ent['end'], ent['label']))
            
            transformed.append((cleaned_text, {"entities": spacy_entities}))
        
        self.stats.invalid_samples += invalid
        
        logger.info(f"   ✓ Valid samples: {len(transformed)}")
        logger.info(f"   ✗ Invalid samples: {invalid}")
        
        return transformed
    
    def transform_all(self, extracted: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform all extracted data.
        
        Args:
            extracted: Dictionary of extracted data
            
        Returns:
            Dictionary of transformed data
        """
        logger.info("\n" + "=" * 50)
        logger.info("🔄 TRANSFORM PHASE")
        logger.info("=" * 50)
        
        transformed = {}
        
        if 'sentiment' in extracted:
            transformed['sentiment'] = self.transform_sentiment_data(extracted['sentiment'])
        
        if 'ner' in extracted:
            transformed['ner'] = self.transform_ner_data(extracted['ner'])
        
        return transformed
    
    # ==================== LOAD ====================
    
    def split_data(
        self, 
        data: List, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[List, List, List]:
        """
        Split data into train/validation/test sets.
        
        Args:
            data: Data to split
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            Tuple of (train, validation, test) sets
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
        
        # Shuffle data
        shuffled = data.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]
        
        logger.info(f"   Split: train={len(train)}, val={len(val)}, test={len(test)}")
        
        return train, val, test
    
    def calculate_stats(self, data: List[Dict]) -> None:
        """
        Calculate dataset statistics.
        
        Args:
            data: Processed data
        """
        self.stats.total_samples = len(data)
        
        # Text lengths
        lengths = [d.get('text_length', len(d.get('text', '').split())) for d in data]
        if lengths:
            self.stats.avg_text_length = sum(lengths) / len(lengths)
            self.stats.min_text_length = min(lengths)
            self.stats.max_text_length = max(lengths)
        
        # Class distribution
        if 'sentiment' in data[0]:
            self.stats.class_distribution = dict(Counter(d['sentiment'] for d in data))
    
    def save_processed_data(
        self, 
        data: Dict[str, Any], 
        output_dir: str = "data/processed"
    ) -> None:
        """
        Save processed data to files.
        
        Args:
            data: Processed data dictionary
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, dataset in data.items():
            if isinstance(dataset, dict) and 'train' in dataset:
                # Save splits separately
                for split_name, split_data in dataset.items():
                    if split_name != 'stats':
                        filepath = output_path / f"{name}_{split_name}.json"
                        with open(filepath, 'w') as f:
                            json.dump(split_data, f, indent=2)
                        logger.info(f"   ✓ Saved: {filepath}")
    
    def load_splits(self, transformed: Dict[str, Any]) -> Dict[str, ProcessedData]:
        """
        Create train/val/test splits for all datasets.
        
        Args:
            transformed: Transformed data
            
        Returns:
            Dictionary of ProcessedData objects
        """
        logger.info("\n" + "=" * 50)
        logger.info("🔄 LOAD PHASE")
        logger.info("=" * 50)
        
        processed = {}
        
        for name, data in transformed.items():
            logger.info(f"\n📊 Processing: {name}")
            
            if isinstance(data, list) and len(data) > 0:
                train, val, test = self.split_data(data)
                
                # Calculate stats
                if isinstance(data[0], dict):
                    self.calculate_stats(data)
                
                processed[name] = {
                    'train': train,
                    'validation': val,
                    'test': test,
                    'stats': {
                        'total': len(data),
                        'train_size': len(train),
                        'val_size': len(val),
                        'test_size': len(test)
                    }
                }
        
        return processed
    
    # ==================== MAIN PIPELINE ====================
    
    def run(self, save_output: bool = True) -> Dict[str, Any]:
        """
        Run the complete ETL pipeline.
        
        Args:
            save_output: Whether to save processed data to files
            
        Returns:
            Processed datasets ready for ML training
        """
        logger.info("\n" + "=" * 60)
        logger.info("🚀 STARTING ETL PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Extract
        extracted = self.extract_all()
        
        # Step 2: Transform
        transformed = self.transform_all(extracted)
        
        # Step 3: Load
        processed = self.load_splits(transformed)
        
        # Save if requested
        if save_output:
            logger.info("\n💾 Saving processed data...")
            self.save_processed_data(processed)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("✅ ETL PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"\n📊 SUMMARY:")
        logger.info(f"   Total datasets processed: {len(processed)}")
        logger.info(f"   Duplicates removed: {self.stats.duplicates_removed}")
        logger.info(f"   Invalid samples: {self.stats.invalid_samples}")
        
        for name, data in processed.items():
            logger.info(f"\n   {name.upper()}:")
            logger.info(f"      Train: {data['stats']['train_size']} samples")
            logger.info(f"      Val:   {data['stats']['val_size']} samples")
            logger.info(f"      Test:  {data['stats']['test_size']} samples")
        
        return processed


# ==================== MAIN ====================

if __name__ == "__main__":
    # Run the ETL pipeline
    pipeline = MedicalDataPipeline(data_dir="data/training")
    processed_data = pipeline.run(save_output=True)
    
    print("\n🎉 ETL Pipeline completed successfully!")
    print("Processed data saved to: data/processed/")
