# app.py
from flask import Flask, request, jsonify, render_template
import json
from medical_nlp import MedicalNLPPipeline

app = Flask(__name__)

# Initialize pipeline on demand instead of at startup
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = MedicalNLPPipeline()
    return pipeline

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_transcript():
    data = request.json
    transcript = data.get('transcript', '')
    
    if not transcript:
        return jsonify({"error": "No transcript provided"}), 400
    
    # Get the pipeline instance (initialized on first use)
    nlp_pipeline = get_pipeline()
    
    # Process the transcript using our pipeline
    medical_summary = nlp_pipeline.summarize_medical_details(transcript)
    sentiment_intent = nlp_pipeline.analyze_patient_sentiment_and_intent(transcript)
    soap_note = nlp_pipeline.generate_soap_note(transcript)
    
    result = {
        "medical_summary": medical_summary,
        "sentiment_intent": sentiment_intent,
        "soap_note": soap_note
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')