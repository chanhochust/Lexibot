import os
import sys
import markdown
import logging
import json
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify, session
import pymongo
from src.rag_chain import build_rag_chain, ask_question
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = "lexibot-ajax-fix-key"

MONGO_URI = os.getenv("MONGO_URI")

# Kết nối MongoDB
try:
    if MONGO_URI:
        client = pymongo.MongoClient(MONGO_URI)
        # Tên Database: lexibot_db, Tên Collection: feedbacks
        db = client.lexibot_db
        feedback_col = db.feedbacks
        print("Đã kết nối thành công tới MongoDB Atlas")
    else:
        print("Chưa có MONGO_URI trong file .env")
        db = None
except Exception as e:
    print(f"Lỗi kết nối MongoDB: {e}")
    db = None

LEXIBOT_CHAIN = {}

def get_chain(model_provider: str):
    global LEXIBOT_CHAIN
    if model_provider not in LEXIBOT_CHAIN:
        LEXIBOT_CHAIN[model_provider] = build_rag_chain(model_provider)
    return LEXIBOT_CHAIN[model_provider]

def simplify_sources(docs):
    simple = []
    if not docs: return simple
    for d in docs:
        path = d.metadata.get("source", "Tài liệu")
        file_name = os.path.basename(str(path))
        section = d.metadata.get("section", "Thông tin chung")
        simple.append({"file": file_name, "section": section})
    
    unique = []
    seen = set()
    for s in simple:
        uid = f"{s['file']}-{s['section']}"
        if uid not in seen:
            unique.append(s)
            seen.add(uid)
    return unique

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", chat_history=session.get("chat_history", []))

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Dữ liệu không hợp lệ"}), 400
            
        question = data.get("question")
        model = data.get("model", "gemini")
        chain = get_chain(model)
        chat_history = session.get("chat_history", [])

        answer_raw, raw_docs = ask_question(chain, question, chat_history)
        
        answer_html = markdown.markdown(answer_raw, extensions=['tables', 'fenced_code'])
        safe_sources = simplify_sources(raw_docs)
        
        # Cập nhật session
        chat_history.append({"role": "user", "content": question})
        chat_history.append({
            "role": "assistant", 
            "content": answer_html, 
            "sources": safe_sources,
            "model": model
        })
        session["chat_history"] = chat_history
        session.modified = True

        return jsonify({
            "answer": answer_html,
            "sources": safe_sources,
            "model": model
        })

    except Exception as e:
        print(f"[ERROR]: {str(e)}", flush=True)
        return jsonify({"error": str(e)}), 500

@app.route("/feedback", methods=["POST"])
def feedback():
    """Lưu phản hồi vào MongoDB"""
    try:
        data = request.get_json()
        
        feedback_doc = {
            "type": data.get("type"), # 'like' hoặc 'dislike'
            "question": data.get("question"),
            "answer": data.get("answer"),
            "model": data.get("model", "unknown"),
            "timestamp": datetime.now(timezone.utc)
        }

        if db is not None:
            result = feedback_col.insert_one(feedback_doc)
            return jsonify({"status": "success", "storage": "mongodb", "id": str(result.inserted_id)})
        else:
            # Fallback nếu DB chưa cấu hình (ghi vào file local để không mất dữ liệu)
            with open("feedback_logs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            return jsonify({"status": "success", "storage": "local_fallback"})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clear", methods=["POST"])
def clear():
    session.pop("chat_history", None)
    return jsonify({"status": "success"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)