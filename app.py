import os
import sys
import markdown
import logging
from flask import Flask, render_template, request, jsonify, session

from src.rag_chain import build_rag_chain, ask_question

app = Flask(__name__)
app.secret_key = "lexibot-ajax-fix-key"

LEXIBOT_CHAIN = None

def get_chain():
    global LEXIBOT_CHAIN
    if LEXIBOT_CHAIN is None:
        LEXIBOT_CHAIN = build_rag_chain()
    return LEXIBOT_CHAIN

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
        chain = get_chain()
        chat_history = session.get("chat_history", [])

        answer_raw, raw_docs = ask_question(chain, question, chat_history)
        
        answer_html = markdown.markdown(answer_raw)
        safe_sources = simplify_sources(raw_docs)
        
        # Cập nhật session
        chat_history.append({"role": "user", "content": question})
        chat_history.append({
            "role": "assistant", 
            "content": answer_html, 
            "sources": safe_sources
        })
        session["chat_history"] = chat_history
        session.modified = True

        return jsonify({
            "answer": answer_html,
            "sources": safe_sources
        })

    except Exception as e:
        print(f"[ERROR]: {str(e)}", flush=True)
        return jsonify({"error": str(e)}), 500

@app.route("/clear", methods=["POST"])
def clear():
    session.pop("chat_history", None)
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)