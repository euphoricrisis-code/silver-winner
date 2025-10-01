from flask import Flask, request, jsonify, Response
import requests
import os
import json
from datetime import datetime

app = Flask(__name__)

# Get NVIDIA API key from environment variable
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '')
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        
        # Extract OpenAI format parameters
        messages = data.get('messages', [])
        model = data.get('model', 'meta/llama-3.1-405b-instruct')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1024)
        stream = data.get('stream', False)
        
        # Convert to NVIDIA NIM format
        nim_payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Add optional parameters if present
        if 'top_p' in data:
            nim_payload['top_p'] = data['top_p']
        if 'frequency_penalty' in data:
            nim_payload['frequency_penalty'] = data['frequency_penalty']
        if 'presence_penalty' in data:
            nim_payload['presence_penalty'] = data['presence_penalty']
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Make request to NVIDIA NIM
        nim_response = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=headers,
            json=nim_payload,
            stream=stream
        )
        
        if stream:
            # Handle streaming response
            def generate():
                for chunk in nim_response.iter_lines():
                    if chunk:
                        yield chunk + b'\n'
            
            return Response(generate(), mimetype='text/event-stream')
        else:
            # Handle non-streaming response
            return jsonify(nim_response.json()), nim_response.status_code
            
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "proxy_error",
                "code": "internal_error"
            }
        }), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models"""
    try:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        nim_response = requests.get(
            f"{NVIDIA_BASE_URL}/models",
            headers=headers
        )
        
        return jsonify(nim_response.json()), nim_response.status_code
    except Exception as e:
        # Return default models if API call fails
        return jsonify({
            "object": "list",
            "data": [
                {"id": "meta/llama-3.1-405b-instruct", "object": "model"},
                {"id": "meta/llama-3.1-70b-instruct", "object": "model"},
                {"id": "meta/llama-3.1-8b-instruct", "object": "model"}
            ]
        })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "nvidia-nim-proxy"})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "NVIDIA NIM to OpenAI API Proxy",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
