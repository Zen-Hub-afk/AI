import os
import json
from flask import Flask, request, jsonify
from google import genai
from flask_cors import CORS 

# --- Configuration and Initialization ---
# NOTE FOR DEPLOYMENT:
# The GEMINI_API_KEY MUST be set as an environment variable 
# in your hosting platform (e.g., Cloud Run, Heroku, etc.).
# This code will fail to run if the key is not set.

try:
    API_KEY = os.environ.get('AIzaSyDPGVVVoBOVgu1AqBerK_t3QVAq7Mf0-cM')
    if not API_KEY:
        # Raise an exception if the key is missing to ensure secure deployment practice.
        # Running locally requires setting the environment variable first.
        raise ValueError("GEMINI_API_KEY environment variable is not set. Please configure it in your deployment environment.")
        
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    # Print error but allow the Flask app to start so the user can see the status page if needed
    print(f"FATAL ERROR: Failed to initialize Gemini Client. Details: {e}")
    client = None

# The model name we will use for text generation with Google Search grounding
MODEL_NAME = "gemini-2.5-flash" 

# --- Flask App Setup ---
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow requests from any origin (e.g., your GitHub Pages URL)
CORS(app) 

# System instruction to guide the model's behavior
SYSTEM_INSTRUCTION = (
    "You are a friendly, helpful, and highly informative AI assistant named RAU-01. "
    "Your core function is to provide answers based on real-time information. "
    "Always use the Google Search tool for grounded, up-to-date information, especially for current events or facts. "
    "Be concise, clear, and encouraging. Your responses must be structured well and easy to read."
    "you MUST use the provided Google Search results to ground your answer and provide citations in the final output. If no relevant information is found, respond accordingly. moreover, to have a type personality that
)

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    """Handles the chat request from the frontend and communicates with the Gemini API."""
    
    # Check if the client was initialized successfully
    if not client:
        return jsonify({'error': 'AI Service Initialization Failed. Check GEMINI_API_KEY configuration.'}), 503

    try:
        # 1. Get the prompt from the JSON body
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Missing prompt in request'}), 400
        
        user_prompt = data['prompt']

        # 2. Configure the API call for search grounding
        tools = [{"google_search": {}}]
        
        config = genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=tools,
        )

        # 3. Call the Gemini API
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[user_prompt],
            config=config,
        )
        
        # 4. Extract text and sources
        generated_text = response.text
        
        # Extract grounding sources for citation display
        sources = []
        grounding_metadata = response.candidates[0].grounding_metadata
        if grounding_metadata and grounding_metadata.grounding_attributions:
            sources = [
                {
                    'uri': attr.web.uri,
                    'title': attr.web.title,
                }
                for attr in grounding_metadata.grounding_attributions
                # Only include sources with both URI and Title for clean citation display
                if attr.web and attr.web.uri and attr.web.title
            ]
        
        # 5. Return the AI response (text and sources) to the frontend
        return jsonify({
            'text': generated_text,
            'sources': sources
        })

    except Exception as e:
        # General error handling for issues like API call failure
        print(f"An unexpected error occurred during API call: {e}")
        return jsonify({'error': f'Internal Server Error. Please try again. Details: {str(e)}'}), 500

if __name__ == '__main__':
    # This is for local testing. In production, the port is usually set by the hosting environment.
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting server on port {port}")
    # Using '0.0.0.0' allows access from outside if firewall permits, which is standard for container deployment
    app.run(host='0.0.0.0', port=port)
