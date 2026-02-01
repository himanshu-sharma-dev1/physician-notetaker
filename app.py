"""
🩺 Physician Notetaker - Streamlit Application

A comprehensive AI-powered medical transcription system with:
- Medical NER extraction with entity highlighting
- Text summarization and structured reports
- Sentiment and intent analysis with journey visualization
- SOAP note generation
- Keyword extraction

Author: Himanshu Sharma
For: Emitrr AI Engineer Intern Assignment
"""

import os
import json
import sys
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.medical_ner import MedicalNERExtractor
from src.summarizer import MedicalSummarizer
from src.sentiment_analyzer import MedicalSentimentAnalyzer
from src.keyword_extractor import MedicalKeywordExtractor
from src.soap_generator import SOAPNoteGenerator

# Try to import annotated text for entity highlighting
try:
    from annotated_text import annotated_text
    ANNOTATED_TEXT_AVAILABLE = True
except ImportError:
    ANNOTATED_TEXT_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Physician Notetaker | AI Medical Transcription",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Sentiment badges */
    .sentiment-anxious {
        background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
        padding: 8px 16px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        padding: 8px 16px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
    }
    .sentiment-reassured {
        background: linear-gradient(135deg, #55efc4, #00b894);
        padding: 8px 16px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
    }
    
    /* Entity tags */
    .entity-symptom {
        background: #ff7675;
        padding: 4px 8px;
        border-radius: 4px;
        margin: 2px;
        display: inline-block;
    }
    .entity-treatment {
        background: #74b9ff;
        padding: 4px 8px;
        border-radius: 4px;
        margin: 2px;
        display: inline-block;
    }
    .entity-diagnosis {
        background: #a29bfe;
        padding: 4px 8px;
        border-radius: 4px;
        margin: 2px;
        display: inline-block;
    }
    
    /* SOAP sections */
    .soap-section {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #4facfe;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


def load_sample_conversation():
    """Load the sample conversation from file."""
    sample_path = Path(__file__).parent / "data" / "sample_conversation.txt"
    if sample_path.exists():
        return sample_path.read_text()
    return """Physician: Good morning. How are you feeling today?

Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.

Physician: I understand you were in a car accident. Can you tell me what happened?

Patient: Yes, it was in September. Another car hit me from behind. I had pain in my neck and back right away.

Physician: Did you seek medical attention?

Patient: Yes, they said it was a whiplash injury. I had to take painkillers and go through ten sessions of physiotherapy.

Physician: Are you still experiencing pain now?

Patient: Just occasional backaches. Nothing like before.

Physician: That's good. Based on your progress, I expect you to make a full recovery within six months."""


def create_sentiment_gauge(sentiment: str, confidence: float):
    """Create a sentiment gauge visualization."""
    color_map = {
        "Anxious": "#ff6b6b",
        "Neutral": "#74b9ff", 
        "Reassured": "#55efc4"
    }
    color = color_map.get(sentiment, "#74b9ff")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Sentiment: {sentiment}", 'font': {'size': 16, 'color': 'white'}},
        number={'suffix': "%", 'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': color},
            'bgcolor': 'rgba(255,255,255,0.1)',
            'borderwidth': 2,
            'bordercolor': 'white',
            'steps': [
                {'range': [0, 33], 'color': 'rgba(255,107,107,0.3)'},
                {'range': [33, 66], 'color': 'rgba(116,185,255,0.3)'},
                {'range': [66, 100], 'color': 'rgba(85,239,196,0.3)'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_sentiment_journey_chart(journey: list):
    """Create a line chart showing sentiment progression."""
    if not journey:
        return None
    
    sentiment_values = {"Anxious": 1, "Neutral": 2, "Reassured": 3}
    
    phases = [j.get("phase", f"Phase {i+1}") for i, j in enumerate(journey)]
    values = [sentiment_values.get(j.get("sentiment", "Neutral"), 2) for j in journey]
    colors = [
        "#ff6b6b" if v == 1 else "#74b9ff" if v == 2 else "#55efc4" 
        for v in values
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=phases,
        y=values,
        mode='lines+markers',
        line=dict(color='#4facfe', width=3),
        marker=dict(size=12, color=colors, line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor='rgba(79, 172, 254, 0.1)'
    ))
    
    fig.update_layout(
        title="Sentiment Journey Through Conversation",
        xaxis_title="Conversation Phase",
        yaxis=dict(
            tickvals=[1, 2, 3],
            ticktext=["Anxious", "Neutral", "Reassured"],
            range=[0.5, 3.5]
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def render_entity_tags(entities: dict):
    """Render entity tags with colors."""
    st.markdown("### 🏷️ Extracted Entities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💊 Symptoms**")
        symptoms = entities.get("symptoms", [])
        if symptoms:
            for s in symptoms[:5]:
                text = s.get("text", s) if isinstance(s, dict) else s
                st.markdown(f'<span class="entity-symptom">{text}</span>', unsafe_allow_html=True)
        else:
            st.info("No symptoms detected")
        
        st.markdown("**🏥 Treatments**")
        treatments = entities.get("treatments", [])
        if treatments:
            for t in treatments[:5]:
                text = t.get("text", t) if isinstance(t, dict) else t
                st.markdown(f'<span class="entity-treatment">{text}</span>', unsafe_allow_html=True)
        else:
            st.info("No treatments detected")
    
    with col2:
        st.markdown("**📋 Diagnoses**")
        diagnoses = entities.get("diagnoses", [])
        if diagnoses:
            for d in diagnoses[:3]:
                text = d.get("text", d) if isinstance(d, dict) else d
                st.markdown(f'<span class="entity-diagnosis">{text}</span>', unsafe_allow_html=True)
        else:
            st.info("No diagnosis detected")
        
        st.markdown("**✅ Prognosis**")
        prognosis = entities.get("prognosis", [])
        if prognosis:
            for p in prognosis[:3]:
                text = p.get("text", p) if isinstance(p, dict) else p
                st.success(text)
        else:
            st.info("No prognosis detected")


def render_soap_note(soap: dict):
    """Render SOAP note in clinical format."""
    st.markdown("### 📋 SOAP Note")
    
    sections = [
        ("S - Subjective", soap.get("Subjective", {}), "🗣️"),
        ("O - Objective", soap.get("Objective", {}), "🔍"),
        ("A - Assessment", soap.get("Assessment", {}), "📊"),
        ("P - Plan", soap.get("Plan", {}), "📝")
    ]
    
    for title, content, icon in sections:
        with st.expander(f"{icon} {title}", expanded=True):
            if isinstance(content, dict):
                for key, value in content.items():
                    st.markdown(f"**{key.replace('_', ' ')}:** {value}")
            else:
                st.write(content)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown("# 🩺 Physician Notetaker")
    st.markdown("*AI-Powered Medical Transcription & Analysis System*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        st.info("🚀 **100% Local ML** - No API keys required!")
        
        st.markdown("---")
        st.markdown("### 📚 Sample Data")
        
        if st.button("📄 Load Sample Conversation"):
            st.session_state['transcript'] = load_sample_conversation()
            st.success("Sample loaded!")
        
        # Load test cases
        test_cases_dir = Path(__file__).parent / "data" / "test_cases"
        if test_cases_dir.exists():
            test_files = list(test_cases_dir.glob("*.txt"))
            if test_files:
                st.markdown("**Test Cases:**")
                for test_file in test_files:
                    if st.button(f"📝 {test_file.stem.replace('_', ' ').title()}"):
                        st.session_state['transcript'] = test_file.read_text()
                        st.success(f"Loaded: {test_file.stem}")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This application analyzes medical transcripts to extract:
        - Patient symptoms & treatments
        - Diagnosis & prognosis
        - Sentiment & intent
        - Structured SOAP notes
        
        **Built for:** Emitrr AI Engineer Intern Assignment
        
        **Author:** Himanshu Sharma
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Analyze", 
        "📊 Results", 
        "📋 SOAP Note",
        "💡 How It Works"
    ])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📝 Enter Transcript")
            transcript = st.text_area(
                "Paste your physician-patient conversation here:",
                value=st.session_state.get('transcript', ''),
                height=400,
                placeholder="Physician: How are you feeling today?\nPatient: I've been experiencing..."
            )
            st.session_state['transcript'] = transcript
        
        with col2:
            st.markdown("### 🎯 Analysis Options")
            
            run_ner = st.checkbox("Named Entity Recognition", value=True)
            run_summary = st.checkbox("Medical Summary", value=True)
            run_sentiment = st.checkbox("Sentiment Analysis", value=True)
            run_keywords = st.checkbox("Keyword Extraction", value=True)
            run_soap = st.checkbox("SOAP Note Generation", value=True)
            
            st.markdown("---")
            
            analyze_btn = st.button("🚀 Analyze Transcript", type="primary", use_container_width=True)
        
        if analyze_btn and transcript:
            with st.spinner("🔄 Analyzing transcript..."):
                try:
                    results = {}
                    
                    if run_ner:
                        with st.status("Extracting medical entities..."):
                            ner = MedicalNERExtractor()
                            entities = ner.extract(transcript)
                            entities = ner.handle_ambiguous_data(entities, transcript)
                            results['entities'] = entities.to_dict()
                            st.write("✅ NER complete")
                    
                    if run_summary:
                        with st.status("Generating summary..."):
                            summarizer = MedicalSummarizer()  # 100% local
                            summary = summarizer.summarize(transcript)
                            results['summary'] = summary.to_dict()
                            st.write("✅ Summary complete")
                    
                    if run_sentiment:
                        with st.status("Analyzing sentiment..."):
                            analyzer = MedicalSentimentAnalyzer()
                            sentiment = analyzer.analyze(transcript)
                            results['sentiment'] = sentiment.to_dict()
                            st.write("✅ Sentiment analysis complete")
                    
                    if run_keywords:
                        with st.status("Extracting keywords..."):
                            extractor = MedicalKeywordExtractor()
                            keywords = extractor.extract(transcript)
                            results['keywords'] = keywords.to_dict()
                            st.write("✅ Keyword extraction complete")
                    
                    if run_soap:
                        with st.status("Generating SOAP note..."):
                            generator = SOAPNoteGenerator()  # 100% local
                            soap = generator.generate(transcript)
                            results['soap'] = soap.to_dict()
                            st.write("✅ SOAP note complete")
                    
                    st.session_state['results'] = results
                    st.success("✅ Analysis complete! Check the Results tab.")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
    
    with tab2:
        if 'results' in st.session_state:
            results = st.session_state['results']
            
            # Summary section
            if 'summary' in results:
                st.markdown("### 📋 Medical Summary")
                summary = results['summary']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Patient", summary.get("Patient_Name", "Unknown"))
                with col2:
                    st.metric("Diagnosis", summary.get("Diagnosis", "N/A"))
                with col3:
                    st.metric("Status", summary.get("Current_Status", "N/A")[:30])
                
                with st.expander("📄 Full Summary JSON"):
                    st.json(summary)
            
            # Entities section
            if 'entities' in results:
                render_entity_tags(results['entities'])
            
            # Sentiment section
            if 'sentiment' in results:
                st.markdown("### 🎭 Sentiment Analysis")
                sentiment_data = results['sentiment']
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    sentiment = sentiment_data.get("Sentiment", "Neutral")
                    confidence = sentiment_data.get("Confidence", 0.5)
                    fig = create_sentiment_gauge(sentiment, confidence)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(f"**Intent:** {sentiment_data.get('Intent', 'Unknown')}")
                
                with col2:
                    details = sentiment_data.get("Details", {})
                    journey = details.get("sentiment_journey", [])
                    
                    if journey:
                        fig = create_sentiment_journey_chart(journey)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            
            # Keywords section
            if 'keywords' in results:
                st.markdown("### 🔑 Key Terms")
                keywords_data = results['keywords']
                
                keywords = keywords_data.get("keywords", [])
                if keywords:
                    keyword_df = [{"Term": k["term"], "Score": k["score"]} for k in keywords[:10]]
                    st.dataframe(keyword_df, use_container_width=True)
                
                phrases = keywords_data.get("key_phrases", [])
                if phrases:
                    st.markdown("**Key Phrases:**")
                    st.write(", ".join(phrases[:10]))
            
            # Export section
            st.markdown("---")
            st.markdown("### 💾 Export Results")
            
            col1, col2 = st.columns(2)
            with col1:
                json_str = json.dumps(results, indent=2)
                st.download_button(
                    "📥 Download JSON",
                    json_str,
                    "analysis_results.json",
                    "application/json"
                )
            with col2:
                if 'soap' in results:
                    soap_text = json.dumps(results['soap'], indent=2)
                    st.download_button(
                        "📥 Download SOAP Note",
                        soap_text,
                        "soap_note.json",
                        "application/json"
                    )
        else:
            st.info("👆 Enter a transcript and click 'Analyze' to see results here.")
    
    with tab3:
        if 'results' in st.session_state and 'soap' in st.session_state['results']:
            render_soap_note(st.session_state['results']['soap'])
        else:
            st.info("👆 Run analysis with SOAP Note enabled to see results here.")
    
    with tab4:
        st.markdown("""
        ### 🧠 How the Pipeline Works
        
        **100% LOCAL ML** - No external API dependencies. All models run on your machine!
        
        #### 1. Named Entity Recognition (NER)
        - **Technology:** spaCy with custom medical entity patterns
        - **Extracts:** Symptoms, Treatments, Diagnoses, Prognosis, Body Parts
        - **Features:** Confidence scoring, context extraction, ambiguity handling
        
        #### 2. Text Summarization
        - **Technology:** Rule-based extraction + optional HuggingFace BART
        - **Output:** Structured medical report in JSON format
        - **Includes:** Patient name, symptoms, diagnosis, treatment, current status, prognosis
        
        #### 3. Sentiment & Intent Analysis
        - **Technology:** Fine-tuned DistilBERT (trainable via Colab notebooks)
        - **Sentiment Classes:** Anxious, Neutral, Reassured
        - **Intent Detection:** Seeking reassurance, Reporting symptoms, Expressing concern, etc.
        - **Feature:** Sentiment journey tracking across conversation phases
        
        #### 4. Keyword Extraction
        - **Technology:** TF-IDF + domain-specific medical vocabulary
        - **Output:** Ranked keywords, medical terms, key phrases
        
        #### 5. SOAP Note Generation
        - **Technology:** Rule-based extraction with NER integration
        - **Output:** Structured note with Subjective, Objective, Assessment, Plan sections
        
        ---
        
        ### 📊 Technical Details
        
        | Component | Model/Method | Purpose |
        |-----------|-------------|---------|
        | NER | spaCy + PhraseMatcher | Entity extraction |
        | Summarization | Rule-based + BART | Intelligent extraction |
        | Sentiment | DistilBERT (fine-tuned) | Emotion classification |
        | Keywords | scikit-learn TF-IDF | Term importance |
        | SOAP | Rule-based + NER | Clinical documentation |
        
        ---
        
        ### 🔐 Privacy & Security
        
        ✅ **No external API calls** - All processing happens locally
        ✅ **No data leaves your machine** - Complete privacy
        ✅ **Trainable models** - Fine-tune with your own data
        
        In production, also ensure:
        - HIPAA compliance for patient data
        - Data encryption at rest and in transit
        - Audit logging for all operations
        """)


if __name__ == "__main__":
    main()
