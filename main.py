from flask import Flask, request, jsonify, send_from_directory
import requests
import docx
import os

app = Flask(__name__, static_folder='static')

API_KEY = "AIzaSyCAw69Ur3JhFx_iuy8m-wa5VTWXsbXfV6o"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Load and chunk the policy doc
def extract_text_from_docx(path="policy.docx"):
    if not os.path.exists(path):
        return ""
    try:
        doc = docx.Document(path)
    except Exception:
        return ""
    texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(texts)

def chunk_text(text, max_len=1500):
    paragraphs = text.split('\n')
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_len:
            current += para + "\n"
        else:
            chunks.append(current.strip())
            current = para + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

POLICY_TEXT = extract_text_from_docx()
POLICY_CHUNKS = chunk_text(POLICY_TEXT)

# Simple similarity by keyword match (improve later with embeddings)
def find_best_chunk(question):
    question_words = set(question.lower().split())
    best_chunk = ""
    best_score = 0
    for chunk in POLICY_CHUNKS:
        chunk_words = set(chunk.lower().split())
        score = len(question_words.intersection(chunk_words))
        if score > best_score:
            best_score = score
            best_chunk = chunk
    # If no good chunk found, fallback to full doc
    return best_chunk if best_chunk else POLICY_TEXT

# Chat history per user (simple in-memory)
chat_sessions = {}

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant who ONLY answers questions based on the provided policy document excerpt. "
    "If information is not found in the excerpt, respond politely with: "
    "'Sorry, that information is not available in the policy document.' "
    "Be clear, concise, and professional."
)

def build_prompt(history, question, context):
    prompt = SYSTEM_INSTRUCTIONS + "\n\n"
    prompt += "Policy Document excerpt:\n\"\"\"\n" + context + "\n\"\"\"\n\n"
    prompt += "Conversation history:\n"
    for h in history:
        prompt += f"{h['role'].capitalize()}: {h['content']}\n"
    prompt += f"User: {question}\nAssistant:"
    return prompt

def call_gemini_api(prompt):
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": API_KEY,
    }
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    response = requests.post(API_URL, headers=headers, json=data)
    if response.status_code != 200:
        return None, f"API error {response.status_code}: {response.text}"
    result = response.json()
    try:
        answer = result['candidates'][0]['content']['parts'][0]['text']
    except Exception:
        answer = None
    return answer, None

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = request.remote_addr  # replace with better session id if needed
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    # Initialize chat history for user
    if user_id not in chat_sessions:
        chat_sessions[user_id] = []

    # Find best chunk matching question
    context = find_best_chunk(question)

    # Build prompt with instructions, context, history, current question
    prompt = build_prompt(chat_sessions[user_id], question, context)

    # Call Gemini API
    answer, error = call_gemini_api(prompt)
    if error:
        return jsonify({"error": error}), 500

    if not answer:
        answer = "Sorry, that information is not available in the policy document."

    # Update chat history
    chat_sessions[user_id].append({"role": "user", "content": question})
    chat_sessions[user_id].append({"role": "assistant", "content": answer})

    return jsonify({"answer": answer.strip()})

if __name__ == '__main__':
    app.run(debug=True)
