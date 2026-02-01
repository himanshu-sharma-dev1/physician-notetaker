"""
SOAP Note Generation Module
============================
Generates structured SOAP (Subjective, Objective, Assessment, Plan) notes 
from medical transcripts.

100% LOCAL - No external API dependencies.

SOAP Note Format:
- Subjective: Patient's chief complaint and history
- Objective: Physical examination findings  
- Assessment: Diagnosis and severity
- Plan: Treatment plan and follow-up
"""

import re
import json
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SubjectiveSection:
    """Subjective section of SOAP note."""
    chief_complaint: str
    history_of_present_illness: str
    
    def to_dict(self) -> Dict:
        return {
            "Chief_Complaint": self.chief_complaint,
            "History_of_Present_Illness": self.history_of_present_illness
        }


@dataclass
class ObjectiveSection:
    """Objective section of SOAP note."""
    physical_exam: str
    observations: str
    
    def to_dict(self) -> Dict:
        return {
            "Physical_Exam": self.physical_exam,
            "Observations": self.observations
        }


@dataclass
class AssessmentSection:
    """Assessment section of SOAP note."""
    diagnosis: str
    severity: str
    
    def to_dict(self) -> Dict:
        return {
            "Diagnosis": self.diagnosis,
            "Severity": self.severity
        }


@dataclass
class PlanSection:
    """Plan section of SOAP note."""
    treatment: str
    follow_up: str
    
    def to_dict(self) -> Dict:
        return {
            "Treatment": self.treatment,
            "Follow_Up": self.follow_up
        }


@dataclass
class SOAPNote:
    """Complete SOAP note."""
    subjective: SubjectiveSection
    objective: ObjectiveSection
    assessment: AssessmentSection
    plan: PlanSection
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format (assignment output)."""
        return {
            "Subjective": self.subjective.to_dict(),
            "Objective": self.objective.to_dict(),
            "Assessment": self.assessment.to_dict(),
            "Plan": self.plan.to_dict()
        }
    
    def to_clinical_format(self) -> str:
        """Convert to readable clinical format."""
        output = []
        output.append("=" * 60)
        output.append("SOAP NOTE")
        output.append("=" * 60)
        output.append("")
        
        output.append("SUBJECTIVE:")
        output.append(f"  Chief Complaint: {self.subjective.chief_complaint}")
        output.append(f"  HPI: {self.subjective.history_of_present_illness}")
        output.append("")
        
        output.append("OBJECTIVE:")
        output.append(f"  Physical Exam: {self.objective.physical_exam}")
        output.append(f"  Observations: {self.objective.observations}")
        output.append("")
        
        output.append("ASSESSMENT:")
        output.append(f"  Diagnosis: {self.assessment.diagnosis}")
        output.append(f"  Severity: {self.assessment.severity}")
        output.append("")
        
        output.append("PLAN:")
        output.append(f"  Treatment: {self.plan.treatment}")
        output.append(f"  Follow-up: {self.plan.follow_up}")
        output.append("")
        output.append("=" * 60)
        
        return "\n".join(output)


class SOAPNoteGenerator:
    """
    Generate structured SOAP notes from medical transcripts.
    
    100% LOCAL - Uses rule-based extraction with NER integration.
    No external API dependencies.
    """
    
    # Medical keywords for section extraction
    SYMPTOM_KEYWORDS = [
        "pain", "ache", "discomfort", "hurt", "sore", "stiff", "swelling", 
        "fatigue", "tired", "weak", "nausea", "dizziness", "headache",
        "fever", "cough", "shortness of breath", "numbness", "tingling"
    ]
    
    EXAM_KEYWORDS = [
        "range of motion", "tenderness", "swelling", "reflexes", 
        "blood pressure", "heart rate", "temperature", "examination"
    ]
    
    TREATMENT_KEYWORDS = [
        "physiotherapy", "physical therapy", "medication", "surgery",
        "painkillers", "antibiotics", "therapy", "treatment", "prescribed",
        "injection", "exercise", "rest", "ice", "compression", "rehabilitation"
    ]
    
    def __init__(self, ner_extractor=None):
        """
        Initialize SOAP generator.
        
        Args:
            ner_extractor: Optional MedicalNERExtractor for entity extraction
        """
        self.ner = ner_extractor
    
    def generate(self, transcript: str) -> SOAPNote:
        """
        Generate SOAP note from transcript.
        Uses intelligent rule-based extraction.
        
        Args:
            transcript: Physician-patient conversation
            
        Returns:
            SOAPNote object
        """
        # Extract entities if NER is available
        entities = {}
        if self.ner:
            try:
                entities = self.ner.extract_entities(transcript)
            except Exception:
                pass
        
        return SOAPNote(
            subjective=self._extract_subjective(transcript, entities),
            objective=self._extract_objective(transcript, entities),
            assessment=self._extract_assessment(transcript, entities),
            plan=self._extract_plan(transcript, entities)
        )
    
    def _extract_subjective(self, transcript: str, entities: Dict) -> SubjectiveSection:
        """Extract subjective section from transcript."""
        
        # Extract chief complaint from patient statements
        chief_complaint = self._extract_chief_complaint(transcript)
        
        # Extract history of present illness
        history = self._extract_history(transcript)
        
        # Enhance with NER entities if available
        if entities.get('symptoms'):
            symptoms_text = ', '.join([e['text'] for e in entities['symptoms'][:3]])
            if symptoms_text and chief_complaint == "Not documented":
                chief_complaint = symptoms_text
        
        return SubjectiveSection(
            chief_complaint=chief_complaint,
            history_of_present_illness=history
        )
    
    def _extract_chief_complaint(self, transcript: str) -> str:
        """Extract the main complaint from patient."""
        patterns = [
            # "I have/had..." patterns
            r"(?:I |I've been |I am )(?:have|had|having|experiencing|getting)?\s*(.+?)(?:\.|,|for|since|$)",
            # "My X hurts/aches" patterns
            r"[mM]y\s+(.+?)\s+(?:hurt|hurts|is hurting|ache|aches|is aching)",
            # Accident/injury patterns
            r"(?:had a|was in a|car|vehicle)\s+(accident|crash|collision).+?(?:\.|$)",
            # Body part + symptom
            r"(neck|back|head|chest|arm|leg|knee|shoulder).+?(?:pain|ache|hurt)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                complaint = match.group(1) if match.lastindex else match.group(0)
                complaint = complaint.strip()
                if len(complaint) > 3:
                    return complaint[:150]
        
        return "Not documented"
    
    def _extract_history(self, transcript: str) -> str:
        """Extract history of present illness from patient dialogue."""
        # Get all patient statements
        patient_lines = []
        lines = transcript.split('\n')
        
        for line in lines:
            # Check for patient dialogue markers
            if re.match(r'^patient[:：]', line, re.IGNORECASE):
                content = re.sub(r'^patient[:：]\s*', '', line, flags=re.IGNORECASE)
                if content.strip():
                    patient_lines.append(content.strip())
        
        # If no explicit markers, look for first-person statements
        if not patient_lines:
            first_person = re.findall(r"I (?:have|had|'ve|am|was|'m).+?(?:\.|$)", transcript)
            patient_lines = [line.strip() for line in first_person[:3]]
        
        if patient_lines:
            history = ' '.join(patient_lines[:3])
            return history[:400]
        
        return "Not documented"
    
    def _extract_objective(self, transcript: str, entities: Dict) -> ObjectiveSection:
        """Extract objective section from transcript."""
        
        # Look for physical exam findings
        physical_exam = self._extract_exam_findings(transcript)
        
        # Look for general observations
        observations = self._extract_observations(transcript)
        
        return ObjectiveSection(
            physical_exam=physical_exam,
            observations=observations
        )
    
    def _extract_exam_findings(self, transcript: str) -> str:
        """Extract physical examination findings."""
        patterns = [
            # Explicit exam mentions
            r"(?:physical exam|examination|exam)(?:ination)?[:：]?\s*(.+?)(?:\.|$)",
            # Range of motion
            r"((?:full|limited|decreased|normal)\s+range of (?:motion|movement).+?)(?:\.|$)",
            # Tenderness/swelling findings
            r"((?:no|some|mild|moderate|severe)\s+(?:tenderness|swelling|inflammation).+?)(?:\.|$)",
            # Vital signs
            r"(blood pressure|heart rate|temperature|pulse)[:：\s]+(.+?)(?:\.|,|$)",
        ]
        
        findings = []
        for pattern in patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    finding = ' '.join(match).strip()
                else:
                    finding = match.strip()
                if finding and len(finding) > 3:
                    findings.append(finding)
        
        if findings:
            return '; '.join(findings[:3])[:300]
        
        return "Physical examination findings not documented"
    
    def _extract_observations(self, transcript: str) -> str:
        """Extract general observations about patient."""
        patterns = [
            # Patient appearance
            r"(?:patient\s+)?(?:look|looks|appears?|appears to be|seems?)\s+(.+?)(?:\.|,|$)",
            # Health status
            r"(?:patient\s+)?(?:is|seems)\s+(?:in\s+)?(good|stable|fair|poor)\s+(?:health|condition|shape)",
            # Progress indicators
            r"(?:doing|feeling|getting)\s+(better|worse|well|improved)",
        ]
        
        observations = []
        for pattern in patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                obs = match.group(0).strip()
                observations.append(obs)
        
        if observations:
            return '; '.join(observations[:2])[:200]
        
        return "Patient appears in stable condition"
    
    def _extract_assessment(self, transcript: str, entities: Dict) -> AssessmentSection:
        """Extract assessment section from transcript."""
        
        # Extract diagnosis
        diagnosis = self._extract_diagnosis(transcript, entities)
        
        # Determine severity
        severity = self._determine_severity(transcript)
        
        return AssessmentSection(
            diagnosis=diagnosis,
            severity=severity
        )
    
    def _extract_diagnosis(self, transcript: str, entities: Dict) -> str:
        """Extract diagnosis from transcript."""
        
        # First check NER entities
        if entities.get('diagnoses'):
            return entities['diagnoses'][0]['text'].title()
        
        # Pattern-based extraction
        patterns = [
            # Direct diagnosis statements
            r"(?:diagnosis|diagnosed with|it was|it's a?|confirms?)\s+(?:a\s+)?([A-Za-z\s]+?)(?:\.|,|and|$)",
            # Common medical conditions
            r"\b(whiplash injury|whiplash|fracture|sprain|strain|concussion|contusion)\b",
            r"\b(arthritis|diabetes|hypertension|asthma|bronchitis|pneumonia)\b",
            r"\b(anxiety|depression|migraine|vertigo|GERD|IBS)\b",
            # Injury types
            r"(\w+\s+(?:injury|syndrome|disease|disorder|condition))",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                diagnosis = match.group(1).strip()
                if len(diagnosis) > 2:
                    return diagnosis.title()
        
        return "Diagnosis pending further evaluation"
    
    def _determine_severity(self, transcript: str) -> str:
        """Determine severity level from transcript."""
        text_lower = transcript.lower()
        
        # Check for severity indicators
        severe_indicators = ["severe", "serious", "significant", "critical", "emergency", "intense", "unbearable"]
        moderate_indicators = ["moderate", "considerable", "noticeable", "persistent"]
        mild_indicators = ["mild", "slight", "minor", "occasional", "improving", "better"]
        
        for word in severe_indicators:
            if word in text_lower:
                return "Severe"
        
        for word in moderate_indicators:
            if word in text_lower:
                return "Moderate"
        
        for word in mild_indicators:
            if word in text_lower:
                return "Mild, improving"
        
        # Check for progression
        if any(word in text_lower for word in ["improving", "better", "recovery", "healing"]):
            return "Mild to moderate, improving"
        
        return "Moderate"
    
    def _extract_plan(self, transcript: str, entities: Dict) -> PlanSection:
        """Extract plan section from transcript."""
        
        # Extract treatment plan
        treatment = self._extract_treatment(transcript, entities)
        
        # Extract follow-up instructions
        follow_up = self._extract_followup(transcript)
        
        return PlanSection(
            treatment=treatment,
            follow_up=follow_up
        )
    
    def _extract_treatment(self, transcript: str, entities: Dict) -> str:
        """Extract treatment plan from transcript."""
        
        # Check NER entities first
        treatments = []
        if entities.get('treatments'):
            treatments = [e['text'] for e in entities['treatments'][:3]]
        
        # Pattern-based extraction
        patterns = [
            # Direct treatment mentions
            r"(?:recommend|prescribe|start|continue|try)\s+(.+?)(?:\.|,|for|$)",
            # Sessions/therapy mentions
            r"(\d+\s+(?:sessions?|treatments?)\s+of\s+[\w\s]+)",
            # Medication mentions
            r"(?:take|taking|given|prescribed)\s+([\w\s]+\d*\s*(?:mg|ml)?)",
            # Therapy types
            r"(physiotherapy|physical therapy|occupational therapy|speech therapy)",
            r"(rest|ice|compression|elevation|RICE)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str) and 3 < len(match) < 100:
                    treatments.append(match.strip())
        
        if treatments:
            return '; '.join(list(set(treatments))[:4])[:300]
        
        return "Continue current treatment as appropriate"
    
    def _extract_followup(self, transcript: str) -> str:
        """Extract follow-up instructions from transcript."""
        patterns = [
            # Explicit follow-up mentions
            r"(?:follow[- ]?up|come back|return|see me|schedule)(?:\s+(?:in|within|after))?\s*(.+?)(?:\.|$)",
            # Time-based instructions
            r"(?:in|within|after)\s+(\d+\s+(?:days?|weeks?|months?))",
            # Conditional follow-up
            r"(?:if|when|should)\s+(.+?)(?:come back|return|call|contact)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                followup = match.group(0).strip()
                if len(followup) > 5:
                    return followup[:200]
        
        return "Return if symptoms worsen or persist"


# Convenience function
def generate_soap_note(transcript: str) -> Dict:
    """
    Quick function to generate SOAP note.
    100% LOCAL - no API keys required.
    
    Args:
        transcript: Medical conversation
        
    Returns:
        Dictionary with SOAP note structure
    """
    generator = SOAPNoteGenerator()
    soap = generator.generate(transcript)
    return soap.to_dict()


if __name__ == "__main__":
    sample = """
    Physician: Good morning, Ms. Johnson. How are you feeling today?
    
    Patient: Good morning, doctor. I had a car accident about four weeks ago. 
    My neck and back have been hurting a lot since then.
    
    Physician: I'm sorry to hear that. Can you describe the pain?
    
    Patient: It was really severe at first. They said it was a whiplash injury.
    I've been taking painkillers and had ten physiotherapy sessions.
    
    Physician: That's good. Are you feeling any better now?
    
    Patient: Yes, the pain is much better now. I only have occasional back pain.
    
    Physician: I'm pleased to hear that. Based on your progress, I'd expect you 
    to make a full recovery within six months. Continue with the exercises.
    """
    
    print("Testing LOCAL SOAP Generator (No API Required)\n")
    print("=" * 50)
    
    result = generate_soap_note(sample)
    print(json.dumps(result, indent=2))
    
    print("\n" + "=" * 50)
    print("\nClinical Format:")
    print("=" * 50)
    
    generator = SOAPNoteGenerator()
    soap = generator.generate(sample)
    print(soap.to_clinical_format())
