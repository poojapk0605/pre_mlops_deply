#!/usr/bin/env python
import json
import sys
import time
import uuid
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from main import ask_question, clean_answer  # Import your RAG system

from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize once at startup
print("Initializing ASK NEU RAG System...", file=sys.stderr)
# No need to initialize anything here - RAG agent is created on first use
print("ASK NEU System initialized and ready to serve requests", file=sys.stderr)

# Simple in-memory feedback storage
feedback_store = {}

from flask import Response
import time

@app.route('/query', methods=['POST'])
def query_handler():
    data = request.json
    query = data.get('query', '')
    namespace = data.get('namespace', 'default')
    search_mode = data.get('search_mode', 'direct')
    feedback_id = data.get('feedback_id', str(uuid.uuid4()))

    print(f"Processing query: {query} | namespace: {namespace} | search_mode: {search_mode}", file=sys.stderr)

    try:
        result = ask_question(
            question=query,
            namespace=namespace,
            search_mode=search_mode,
            verbose=False
        )

        clean_result = clean_answer(result)

        response = {
            'answer': clean_result.get('answer', ''),
            'sources': clean_result.get('sources', ''),
            'query_id': feedback_id,
            'processing_time': result.get('processing_time', {}).get('total', 0),
            'search_mode': search_mode
        }

        print("✅ Final response:", response, file=sys.stderr)
        return jsonify(response)

    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500


@app.route('/feedback', methods=['POST'])
def store_feedback():
    data = request.json
    if not data or 'query_id' not in data or 'rating' not in data:
        return jsonify({'error': 'query_id and rating are required'}), 400
    
    try:
        query_id = data['query_id']
        rating = data['rating']  # 'positive' or 'negative'
        feedback_text = data.get('feedback_text', '')
        
        # Store the feedback
        feedback_store[query_id] = {
            'rating': rating,
            'feedback_text': feedback_text,
            'timestamp': time.time()
        }
        
        print(f"Stored feedback for query {query_id}: {rating}", file=sys.stderr)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error storing feedback: {str(e)}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """Simple health check endpoint for monitoring"""
    return jsonify({'status': 'ok', 'service': 'ASK NEU Python Service'})

if __name__ == '__main__':
    # Run the Flask app on port 5001 (different from Node.js)
   port = int(os.environ.get("PORT", 8080))
   app.run(host='0.0.0.0', port=port)
