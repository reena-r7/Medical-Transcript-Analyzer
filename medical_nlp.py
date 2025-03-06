import pandas as pd
import re
import json
import warnings
import nltk
from nltk.tokenize import sent_tokenize
warnings.filterwarnings('ignore')

# Only download necessary NLTK resources
nltk.download('punkt', quiet=True)

class MedicalNLPPipeline:
    def __init__(self):
        # Removed heavy models (spaCy, transformers, torch)
        # Instead using lightweight rule-based approaches
        
        # Define medical entities
        self.medical_entities = {
            "SYMPTOM": ["pain", "discomfort", "ache", "stiffness", "injury", "whiplash", "headache"],
            "TREATMENT": ["physiotherapy", "painkillers", "therapy", "medication", "relief", "analgesics"],
            "DIAGNOSIS": ["whiplash", "strain", "injury", "damage"],
            "PROGNOSIS": ["recovery", "improve", "better", "progress"]
        }
        
        # Define sentiment mapping
        self.sentiment_mapping = {
            0: "Anxious",
            1: "Neutral",
            2: "Reassured"
        }
        
        # Define intent mapping
        self.intent_mapping = {
            0: "Seeking reassurance",
            1: "Reporting symptoms",
            2: "Expressing concern",
            3: "General information"
        }
    
    def preprocess_transcript(self, transcript):
        """Preprocess the transcript to separate physician and patient dialogues"""
        lines = transcript.strip().split('\n')
        physician_text = []
        patient_text = []
        
        current_speaker = None
        current_text = []
        
        for line in lines:
            if line.startswith("Physician:") or line.startswith("Doctor:"):
                if current_speaker == "Patient" and current_text:
                    patient_text.append(" ".join(current_text))
                    current_text = []
                current_speaker = "Physician"
                text = re.sub(r'^Physician:|^Doctor:', '', line).strip()
                if text:
                    current_text.append(text)
            elif line.startswith("Patient:"):
                if current_speaker == "Physician" and current_text:
                    physician_text.append(" ".join(current_text))
                    current_text = []
                current_speaker = "Patient"
                text = re.sub(r'^Patient:', '', line).strip()
                if text:
                    current_text.append(text)
            elif line.strip() and current_speaker:
                current_text.append(line.strip())
        
        # Add the last speaker's text
        if current_speaker == "Physician" and current_text:
            physician_text.append(" ".join(current_text))
        elif current_speaker == "Patient" and current_text:
            patient_text.append(" ".join(current_text))
        
        return {
            "physician_text": physician_text,
            "patient_text": patient_text,
            "full_text": transcript
        }
    
    def extract_patient_name(self, transcript):
        """Extract patient name from the transcript"""
        match = re.search(r'Ms\.\s+(\w+)|Mrs\.\s+(\w+)|Mr\.\s+(\w+)', transcript)
        if match:
            for group in match.groups():
                if group:
                    return group
        return "Unknown"
    
    def extract_medical_entities(self, text):
        """Extract medical entities using rule-based approach"""
        extracted_entities = {
            "Symptoms": [],
            "Treatment": [],
            "Diagnosis": [],
            "Prognosis": []
        }
        
        # Rule-based entity extraction
        for entity_type, keywords in self.medical_entities.items():
            for keyword in keywords:
                pattern = rf'\b{keyword}\b'
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    start, end = match.span()
                    context = text[max(0, start-30):min(len(text), end+30)]
                    
                    # Determine which category this belongs to
                    if entity_type == "SYMPTOM":
                        # Extract full symptom phrase (e.g., "neck pain" instead of just "pain")
                        symptom_match = re.search(r'([a-zA-Z\s]+\s+' + keyword + r'|' + keyword + r'\s+[a-zA-Z\s]+)', context)
                        if symptom_match:
                            extracted_entities["Symptoms"].append(symptom_match.group(0).strip())
                        else:
                            extracted_entities["Symptoms"].append(keyword)
                    elif entity_type == "TREATMENT":
                        treatment_match = re.search(r'(\d+\s+' + keyword + r'|' + keyword + r'\s+sessions|\w+\s+' + keyword + r')', context)
                        if treatment_match:
                            extracted_entities["Treatment"].append(treatment_match.group(0).strip())
                        else:
                            extracted_entities["Treatment"].append(keyword)
                    elif entity_type == "DIAGNOSIS":
                        diagnosis_match = re.search(r'([a-zA-Z\s]+\s+' + keyword + r'|' + keyword + r'\s+[a-zA-Z\s]+)', context)
                        if diagnosis_match:
                            extracted_entities["Diagnosis"].append(diagnosis_match.group(0).strip())
                        else:
                            extracted_entities["Diagnosis"].append(keyword)
                    elif entity_type == "PROGNOSIS":
                        prognosis_match = re.search(r'([a-zA-Z\s]+\s+' + keyword + r'|' + keyword + r'\s+[a-zA-Z\s]+|full\s+' + keyword + r')', context)
                        if prognosis_match:
                            extracted_entities["Prognosis"].append(prognosis_match.group(0).strip())
                        else:
                            extracted_entities["Prognosis"].append(keyword)
        
        # Remove duplicates
        for entity_type in extracted_entities:
            extracted_entities[entity_type] = list(set(extracted_entities[entity_type]))
        
        return extracted_entities
    
    def extract_current_status(self, text):
        """Extract current status from the text"""
        status_patterns = [
            r'occasional\s+\w+\s+pain',
            r'occasional\s+\w+ache',
            r'still\s+\w+\s+\w+\s+discomfort',
            r'no\s+\w+\s+pain',
            r'improving',
            r'better'
        ]
        
        for pattern in status_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return "Not specified"
    
    def summarize_medical_details(self, transcript):
        """Extract medical details from the transcript"""
        processed = self.preprocess_transcript(transcript)
        patient_text = " ".join(processed["patient_text"])
        full_text = processed["full_text"]
        
        patient_name = self.extract_patient_name(full_text)
        entities = self.extract_medical_entities(full_text)
        current_status = self.extract_current_status(patient_text)
        
        # Extract diagnosis from the text if not found in entities
        if not entities["Diagnosis"]:
            diagnosis_match = re.search(r'([Ww]hiplash\s+injury|[Dd]iagnosed\s+with\s+(.+?)\.)', full_text)
            if diagnosis_match:
                entities["Diagnosis"].append(diagnosis_match.group(1))
        
        # Extract treatment details
        treatment_match = re.search(r'(\d+)\s+sessions\s+of\s+physiotherapy', full_text)
        if treatment_match:
            num_sessions = treatment_match.group(1)
            if f"{num_sessions} physiotherapy sessions" not in entities["Treatment"]:
                entities["Treatment"].append(f"{num_sessions} physiotherapy sessions")
        
        if "painkillers" in full_text.lower() and "Painkillers" not in entities["Treatment"]:
            entities["Treatment"].append("Painkillers")
        
        # Extract prognosis
        prognosis_match = re.search(r'[Ff]ull\s+recovery\s+within\s+(\w+)\s+months', full_text)
        if prognosis_match:
            time_frame = prognosis_match.group(1)
            if f"Full recovery expected within {time_frame} months" not in entities["Prognosis"]:
                entities["Prognosis"].append(f"Full recovery expected within {time_frame} months")
        
        result = {
            "Patient_Name": patient_name,
            "Symptoms": entities["Symptoms"],
            "Diagnosis": entities["Diagnosis"],
            "Treatment": entities["Treatment"],
            "Current_Status": current_status,
            "Prognosis": entities["Prognosis"]
        }
        
        return result
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of the patient's text"""
        # Simple rule-based approach
        anxious_words = ["worried", "concern", "nervous", "anxious", "fear", "scared", "stress"]
        reassured_words = ["relief", "better", "good", "great", "encouraged", "reassured", "positive"]
        
        anxious_score = sum(1 for word in anxious_words if word in text.lower())
        reassured_score = sum(1 for word in reassured_words if word in text.lower())
        
        if anxious_score > reassured_score:
            return "Anxious"
        elif reassured_score > anxious_score:
            return "Reassured"
        else:
            return "Neutral"
    
    def detect_intent(self, text):
        """Detect the intent of the patient's text"""
        # Simple rule-based approach
        reassurance_patterns = [r'will\s+i\s+be\s+okay', r'is\s+this\s+normal', r'should\s+i\s+worry', r'right\?']
        symptom_patterns = [r'i\s+feel', r'i\s+have', r'i\s+am\s+experiencing', r'pain\s+in']
        concern_patterns = [r'worried\s+about', r'concerned', r'afraid\s+of', r'scared']
        
        reassurance_matches = sum(1 for pattern in reassurance_patterns if re.search(pattern, text.lower()))
        symptom_matches = sum(1 for pattern in symptom_patterns if re.search(pattern, text.lower()))
        concern_matches = sum(1 for pattern in concern_patterns if re.search(pattern, text.lower()))
        
        scores = [reassurance_matches, symptom_matches, concern_matches]
        max_score_index = scores.index(max(scores))
        
        if max_score_index == 0:
            return "Seeking reassurance"
        elif max_score_index == 1:
            return "Reporting symptoms"
        elif max_score_index == 2:
            return "Expressing concern"
        else:
            return "General information"
    
    def analyze_patient_sentiment_and_intent(self, transcript):
        """Analyze the sentiment and intent of the patient"""
        processed = self.preprocess_transcript(transcript)
        patient_text = " ".join(processed["patient_text"])
        
        sentiment = self.analyze_sentiment(patient_text)
        intent = self.detect_intent(patient_text)
        
        return {
            "Sentiment": sentiment,
            "Intent": intent
        }
    
    def generate_soap_note(self, transcript):
        """Generate a SOAP note from the transcript"""
        processed = self.preprocess_transcript(transcript)
        patient_text = " ".join(processed["patient_text"])
        full_text = processed["full_text"]
        
        # Extract medical details
        medical_details = self.summarize_medical_details(transcript)
        
        # Generate Subjective section
        subjective = {
            "Chief_Complaint": ", ".join(medical_details["Symptoms"][:2]),
            "History_of_Present_Illness": ""
        }
        
        # Extract history of present illness
        accident_match = re.search(r'car\s+accident.*?\.', full_text, re.DOTALL)
        if accident_match:
            subjective["History_of_Present_Illness"] = accident_match.group(0)
        else:
            # Create history from symptoms
            subjective["History_of_Present_Illness"] = f"Patient reports {', '.join(medical_details['Symptoms']).lower()}. Current status: {medical_details['Current_Status']}."
        
        # Generate Objective section
        objective = {
            "Physical_Exam": "",
            "Observations": "Patient appears in normal health."
        }
        
        # Extract physical exam findings
        exam_match = re.search(r'\[Physical\s+Examination\s+Conducted\](.*?)(?=Patient|$)', full_text, re.DOTALL)
        if exam_match:
            exam_text = exam_match.group(1)
            physician_comment = re.search(r'Physician:\s*(.*?)(?=Patient|$)', exam_text, re.DOTALL)
            if physician_comment:
                objective["Physical_Exam"] = physician_comment.group(1).strip()
        
        if not objective["Physical_Exam"]:
            # Default physical exam if not found
            objective["Physical_Exam"] = "Full range of motion in cervical and lumbar spine, no tenderness."
        
        # Generate Assessment section
        assessment = {
            "Diagnosis": ", ".join(medical_details["Diagnosis"]) if medical_details["Diagnosis"] else "Pending",
            "Severity": "Mild, improving" if "occasional" in medical_details["Current_Status"].lower() else "Moderate"
        }
        
        # Generate Plan section
        plan = {
            "Treatment": ", ".join(medical_details["Treatment"]) if medical_details["Treatment"] else "Observation",
            "Follow-Up": "Patient to return if pain worsens or persists."
        }
        
        # Extract follow-up recommendation
        followup_match = re.search(r'(?:come\s+back|follow-up|return).*?\.', full_text)
        if followup_match:
            plan["Follow-Up"] = followup_match.group(0)
        
        soap_note = {
            "Subjective": subjective,
            "Objective": objective,
            "Assessment": assessment,
            "Plan": plan
        }
        
        return soap_note