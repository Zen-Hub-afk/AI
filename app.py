import os
import requests
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Configuration ---

# **IMPORTANT:** Replace this with your actual Gemini API Key, 
# or preferably, set it as an environment variable.
# Example: export GEMINI_API_KEY='YOUR_API_KEY_HERE'
API_KEY = os.environ.get("GEMINI_API_KEY", "") 

if not API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not found. Using empty string.")

# Model and API Endpoint
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

# Initialize Flask App
app = Flask(__name__)
# Enable CORS so the HTML page running on its own can talk to the server
CORS(app) 

# --- Helper Functions for API Call ---

def get_grounded_content(prompt, max_retries=5):
    """
    Calls the Gemini API with Google Search Grounding enabled.
    Implements exponential backoff for robustness.
    """
    # System Instruction: Define the AI's persona and rules for grounded responses
    system_prompt = "You are a helpful, internet-connected assistant. When responding, you MUST use the provided Google Search results to ground your answer and provide citations in the final output."

    # Payload construction
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        
        # 1. CRITICAL: Enable Google Search Grounding
        "tools": [{"google_search": {}}], 

        # 2. Add System Instruction
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    # Exponential Backoff implementation
    for attempt in range(max_retries):
        try:
            # 3. Make the POST request to the Gemini API
            response = requests.post(
                API_URL, 
                headers={'Content-Type': 'application/json'}, 
                data=json.dumps(payload)
            )
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            
            # Extract generated text
            text = candidate.get('content', {}).get('parts', [{}])[0].get('text', '')

            # Extract grounding sources (citations)
            sources = []
            grounding_metadata = candidate.get('groundingMetadata', {})
            attributions = grounding_metadata.get('groundingAttributions', [])
            
            for attr in attributions:
                web = attr.get('web', {})
                if web.get('uri') and web.get('title'):
                    sources.append({
                        'uri': web['uri'],
                        'title': web['title']
                    })

            return text, sources

        except requests.exceptions.RequestException as e:
            # Handle connection/API errors
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                # Calculate exponential backoff delay
                delay = 2 ** attempt
                time.sleep(delay)
                continue
            else:
                return f"Failed to connect to the Gemini API after {max_retries} attempts.", []

# --- Flask Routes ---

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    """
    API endpoint that receives the user prompt and returns the AI response.
    """
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        print(f"Received prompt: {prompt}")

        # Call the helper function to get the grounded response
        ai_response, sources = get_grounded_content(prompt)

        return jsonify({
            'text': ai_response,
            'sources': sources
        })

    except Exception as e:
        print(f"An internal server error occurred: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Open index.html and ensure the GEMINI_API_KEY is set.")
    # Running on port 5000, accessible by the HTML file

    app.run(port=5000, debug=True)
