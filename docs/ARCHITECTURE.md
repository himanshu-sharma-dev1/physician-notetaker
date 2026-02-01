# System Architecture

## Overview

Physician Notetaker is a modular NLP pipeline for medical conversation analysis.

```
┌─────────────────────────────────────────────────────────────────┐
│                      STREAMLIT UI (app.py)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │  Sentiment   │  │  Medical     │  │   SOAP Note           │ │
│  │  Analyzer    │  │  NER         │  │   Generator           │ │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬───────────┘ │
│         │                 │                      │              │
│         ▼                 ▼                      ▼              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    TRAINED MODELS                           ││
│  │  • DistilBERT (Sentiment) - 91.2% accuracy                 ││
│  │  • spaCy NER (Entities) - 5 entity types                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Descriptions

### 1. Sentiment Analyzer (`src/sentiment_analyzer.py`)

**Purpose:** Classify patient sentiment and detect intent.

| Component | Technology | Output |
|-----------|------------|--------|
| Sentiment | Fine-tuned DistilBERT | Anxious / Neutral / Reassured |
| Intent | Rule-based + ML | Seeking reassurance, Reporting symptoms, etc. |
| Confidence | Softmax probabilities | 0.0 - 1.0 score |

**Model Details:**
- Architecture: `distilbert-base-uncased` fine-tuned
- Training Data: 225 labeled patient statements
- Accuracy: 91.2%

---

### 2. Medical NER (`src/medical_ner.py`)

**Purpose:** Extract medical entities from text.

| Entity Type | Examples |
|-------------|----------|
| SYMPTOM | headache, back pain, nausea |
| TREATMENT | physiotherapy, ibuprofen |
| DIAGNOSIS | whiplash injury, strain |
| BODY_PART | neck, back, spine |
| DURATION | 4 weeks, 6 months |

**Model Details:**
- Architecture: spaCy blank English + custom NER
- Training: 105 labeled examples, 30 epochs
- Final Loss: 8.43

---

### 3. SOAP Generator (`src/soap_generator.py`)

**Purpose:** Generate structured SOAP notes.

```
Input (Transcript)           Output (SOAP Note)
─────────────────────       ────────────────────────
"I had back pain for        {
 4 weeks after the           "Subjective": {
 car accident..."              "Chief_Complaint": "Back pain"
                              },
                              "Objective": {...},
                              "Assessment": {...},
                              "Plan": {...}
                            }
```

---

### 4. Data Pipeline (`src/data_pipeline.py`)

**Purpose:** ETL for training data.

```
Raw Data → Validation → Cleaning → Splitting → Training Ready
              │             │           │
              ▼             ▼           ▼
         Schema Check   Normalize   Train/Val/Test
                        Text        (70/15/15)
```

---

## Data Flow

```
User Input (Text)
       │
       ▼
┌──────────────────────┐
│   Text Preprocessing │
│   • Sentence split   │
│   • Speaker detect   │
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│Sentiment│ │   NER   │
│Analysis │ │Extract  │
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           │
           ▼
┌──────────────────────┐
│   SOAP Generation    │
│   Keyword Extraction │
│   Summary Creation   │
└──────────┬───────────┘
           │
           ▼
    JSON Output
```

---

## Model Training Pipeline

### Sentiment Model (Google Colab)

```python
# notebooks/sentiment_training_fixed.ipynb

1. Load Dataset (225 examples)
2. Preprocess & Tokenize
3. Fine-tune DistilBERT (3 epochs)
4. Evaluate (91.2% accuracy)
5. Export to models/medical_sentiment_production/
```

### NER Model (Google Colab)

```python
# notebooks/ner_training.ipynb

1. Load Dataset (105 examples)
2. Convert to spaCy format
3. Train from scratch (30 epochs)
4. Export to models/medical_ner_model/
```

---

## File Structure

```
physician-notetaker/
├── app.py                    # Streamlit UI
├── src/
│   ├── sentiment_analyzer.py # Sentiment + Intent
│   ├── medical_ner.py        # Entity extraction
│   ├── soap_generator.py     # SOAP notes
│   ├── summarizer.py         # Text summarization
│   ├── keyword_extractor.py  # TF-IDF keywords
│   └── data_pipeline.py      # ETL pipeline
├── models/
│   ├── medical_sentiment_production/  # Trained sentiment
│   └── medical_ner_model/             # Trained NER
├── notebooks/
│   ├── sentiment_training_fixed.ipynb
│   └── ner_training.ipynb
└── docs/
    ├── ARCHITECTURE.md       # This file
    └── QUESTIONS.md          # Theoretical Q&A
```

---

## Performance Metrics

| Model | Metric | Value |
|-------|--------|-------|
| Sentiment | Accuracy | 91.2% |
| Sentiment | F1 (weighted) | 0.91 |
| Sentiment | Precision | 0.93 |
| NER | Training Loss | 8.43 |
| NER | Entity Types | 5 |

---

## Design Decisions

### Why 100% Local ML?

1. **Privacy**: Medical data never leaves the device
2. **Cost**: No API fees or rate limits
3. **Speed**: ~100ms inference time
4. **Control**: Full control over model behavior

### Why DistilBERT for Sentiment?

1. **Efficiency**: 40% smaller than BERT, similar accuracy
2. **Speed**: 60% faster inference
3. **Transfer Learning**: Pre-trained knowledge helps with small datasets

### Why spaCy for NER?

1. **Flexibility**: Train from scratch with custom entities
2. **Speed**: Optimized C++ backend
3. **Integration**: Easy to deploy without heavy dependencies
