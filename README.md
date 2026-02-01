# 🩺 Physician Notetaker

> AI-Powered Medical Transcription, NLP Summarization & Sentiment Analysis System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.29+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![100% Local ML](https://img.shields.io/badge/ML-100%25%20Local-green.svg)]()

---

## 🎯 Overview

Physician Notetaker is a **100% local ML-powered** system that analyzes physician-patient conversations:

- **🔍 Medical NER** - Extract Symptoms, Treatments, Diagnoses, Body Parts, Duration
- **📊 Sentiment Analysis** - Classify as Anxious, Neutral, or Reassured (91.2% accuracy)
- **📝 SOAP Note Generation** - Structured clinical documentation
- **🎯 Intent Detection** - Seeking reassurance, reporting symptoms, etc.
- **🔐 Privacy First** - No external API calls, all processing is local

---

## ✨ Key Features

| Feature | Technology | Accuracy |
|---------|------------|----------|
| Sentiment Analysis | Fine-tuned DistilBERT | **91.2%** |
| Named Entity Recognition | Custom spaCy NER | 5 entity types |
| Keyword Extraction | TF-IDF + Medical vocab | Rule-based |
| SOAP Notes | Rule-based + NER | Structured output |

---

## 🧠 Custom Trained Models

This project includes **custom-trained ML models** - not just API calls!

### Sentiment Classifier (DistilBERT)
- **Architecture**: Fine-tuned `distilbert-base-uncased`
- **Training Data**: 225 labeled patient statements
- **Classes**: Anxious, Neutral, Reassured
- **Test Accuracy**: **91.2%**
- **F1 Score**: 0.91 (weighted)

```
              precision    recall  f1-score   support
     anxious       0.79      1.00      0.88        11
     neutral       1.00      0.92      0.96        12
   reassured       1.00      0.82      0.90        11
    accuracy                           0.91        34
```

### Medical NER (spaCy - Trained from Scratch)
- **Architecture**: Blank English model with custom NER
- **Training Data**: 105 labeled examples
- **Entity Types**: SYMPTOM, TREATMENT, DIAGNOSIS, BODY_PART, DURATION
- **Final Training Loss**: 8.43
- **Epochs**: 30

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/himanshu-sharma-dev1/physician-notetaker.git
cd physician-notetaker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install spaCy for local NER - models included
pip install spacy
```

### Run the Application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

> **Note**: No API keys required! All ML models run locally.

---

## 📁 Project Structure

```
physician-notetaker/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── src/
│   ├── medical_ner.py              # Named Entity Recognition
│   ├── summarizer.py               # Medical text summarization
│   ├── sentiment_analyzer.py       # Sentiment & intent analysis
│   ├── keyword_extractor.py        # Keyword extraction
│   ├── soap_generator.py           # SOAP note generation
│   └── data_pipeline.py            # ETL pipeline
│
├── models/                         # 🔥 TRAINED MODELS
│   ├── medical_sentiment_production/   # Fine-tuned DistilBERT
│   └── medical_ner_model/              # Custom spaCy NER
│
├── notebooks/                      # 🔥 TRAINING NOTEBOOKS
│   ├── sentiment_training_fixed.ipynb  # Sentiment model (Colab)
│   └── ner_training.ipynb              # NER model (Colab)
│
├── data/
│   ├── medical_entities.json       # Medical vocabulary
│   └── training/                   # Training datasets
│
└── docs/
    ├── DEPLOYMENT.md               # Deployment guide
    └── QUESTIONS.md                # Theoretical answers
```

---

## 🔥 Model Training (Google Colab)

### Train Sentiment Model
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

```
notebooks/sentiment_training_fixed.ipynb
```

- Fine-tunes DistilBERT on 225 patient statements
- Achieves 91.2% test accuracy
- ~10 minutes training time with GPU
- Downloads production-ready model

### Train NER Model
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

```
notebooks/ner_training.ipynb
```

- Trains spaCy NER from scratch
- 105 labeled medical entities
- 5 entity types
- ~5 minutes training time

---

## 📊 Sample Usage

### Input
```
Doctor: How are you feeling today?
Patient: I'm really worried about my symptoms. I've been having severe 
headaches for about 2 weeks now.
Doctor: I'll prescribe ibuprofen and we'll do some tests.
Patient: Thank you, doctor. That's a relief.
```

### Output
```json
{
  "Medical_Summary": {
    "Symptoms": ["Headaches"],
    "Duration": "2 weeks", 
    "Treatment": ["Ibuprofen", "Diagnostic tests"],
    "Diagnosis": "Under investigation"
  },
  "Sentiment_Analysis": {
    "Initial": "Anxious",
    "Final": "Reassured",
    "Confidence": "91.2%"
  },
  "Entities": {
    "SYMPTOM": ["headaches"],
    "DURATION": ["2 weeks"],
    "TREATMENT": ["ibuprofen"]
  }
}
```

---

## 🧪 Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI (app.py)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Sentiment  │  │  Medical    │  │    SOAP Note        │ │
│  │  Analyzer   │  │  NER        │  │    Generator        │ │
│  │ (DistilBERT)│  │  (spaCy)    │  │    (Rule-based)     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│         ▼                ▼                     ▼            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              CUSTOM TRAINED MODELS                   │   │
│  │  models/medical_sentiment_production/ (91.2% acc)   │   │
│  │  models/medical_ner_model/ (5 entity types)         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 API Reference

### Sentiment Analysis
```python
from src.sentiment_analyzer import MedicalSentimentAnalyzer

analyzer = MedicalSentimentAnalyzer()  # Auto-loads trained model
result = analyzer.analyze("I'm worried about my symptoms")

print(result.sentiment.sentiment)  # "Anxious"
print(result.sentiment.confidence) # 0.912
```

### Medical NER
```python
from src.medical_ner import MedicalNERExtractor

extractor = MedicalNERExtractor()  # Auto-loads trained model
result = extractor.extract("I have headaches for 3 days")

print(result.symptoms)   # [MedicalEntity(text="headaches", ...)]
print(result.prognosis)  # [MedicalEntity(text="3 days", ...)]
```

---

## ☁️ Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy! (No secrets needed - 100% local ML)

See `docs/DEPLOYMENT.md` for detailed instructions.

---

## 🔐 Privacy & Security

✅ **100% Local Processing** - No data leaves your machine  
✅ **No External APIs** - All models run locally  
✅ **No API Keys Required** - Zero configuration  
✅ **HIPAA-Friendly** - No third-party data sharing  

---

## � Model Performance

| Metric | Sentiment Model | NER Model |
|--------|-----------------|-----------|
| Accuracy | 91.2% | N/A |
| F1 Score | 0.91 | N/A |
| Precision | 0.93 | N/A |
| Recall | 0.91 | N/A |
| Training Loss | - | 8.43 |
| Training Time | ~10 min | ~5 min |

---

## 🎓 Learning Highlights

This project demonstrates:

1. **Transfer Learning** - Fine-tuning DistilBERT for domain-specific classification
2. **Training from Scratch** - Building custom spaCy NER models
3. **ETL Pipeline** - Data cleaning, validation, and preprocessing
4. **Production ML** - Model packaging, versioning, and deployment
5. **100% Local ML** - No API dependencies, privacy-first design

---

## 👨‍💻 Author

**Himanshu Sharma**

Built for Emitrr AI Engineer Intern Assignment

---

## 📄 License

MIT License - see LICENSE file for details
