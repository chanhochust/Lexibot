import os
import sys
import markdown
import logging
import json
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pymongo
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
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
        db = client.lexibot_db
        feedback_col = db.feedbacks
        users_col = db.users
        conversations_col = db.conversations
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

# Auth routes
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Vui lòng nhập đủ thông tin"}), 400
    
    # Kiểm tra độ dài mật khẩu
    if len(password) < 6:
        return jsonify({"error": "Mật khẩu phải có ít nhất 6 ký tự"}), 400

    if db is None: return jsonify({"error": "Lỗi kết nối Database"}), 500

    if users_col.find_one({"username": username}):
        return jsonify({"error": "Tài khoản đã tồn tại"}), 400

    hashed_pw = generate_password_hash(password)
    users_col.insert_one({"username": username, "password": hashed_pw, "created_at": datetime.now()})
    return jsonify({"status": "success"})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if db is None: return jsonify({"error": "Database error"}), 500

    user = users_col.find_one({"username": username})
    if user and check_password_hash(user["password"], password):
        session["user_id"] = str(user["_id"])
        session["username"] = username
        session["current_chat_id"] = None # Reset chat context
        return jsonify({"status": "success"})
    
    return jsonify({"error": "Sai tài khoản hoặc mật khẩu"}), 401

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "success"})

# Main routes
@app.route("/", methods=["GET"])
def index():
    user_id = session.get("user_id")
    chat_history = []
    conversations = []
    
    # Nếu user click vào lịch sử bên sidebar -> có chat_id
    requested_chat_id = request.args.get("chat_id")

    if user_id and db is not None:
        # Lấy danh sách hội thoại của user
        conversations = list(conversations_col.find({"user_id": user_id}).sort("updated_at", -1))
        
        if requested_chat_id:
            # Load nội dung hội thoại cụ thể
            chat = conversations_col.find_one({"_id": ObjectId(requested_chat_id), "user_id": user_id})
            if chat:
                chat_history = chat.get("messages", [])
                session["current_chat_id"] = requested_chat_id
        else:
            # Trang chủ mặc định hoặc sau khi bấm New Chat
            if session.get("current_chat_id"):
                 pass
            else:
                 chat_history = []
    else:
        # Khách: Dùng session cookie
        chat_history = session.get("chat_history", [])

    return render_template("index.html", 
                           chat_history=chat_history, 
                           conversations=conversations,
                           user_id=user_id,
                           username=session.get("username"),
                           current_chat_id=session.get("current_chat_id"))

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        if not data: return jsonify({"error": "Dữ liệu lỗi"}), 400
            
        question = data.get("question")
        model = data.get("model", "gemini")
        chain = get_chain(model)
        
        # Lấy lịch sử để context cho AI
        context_history = [] 
        if session.get("user_id") and session.get("current_chat_id") and db is not None:
             chat_doc = conversations_col.find_one({"_id": ObjectId(session["current_chat_id"])})
             if chat_doc: context_history = chat_doc.get("messages", [])
        else:
             context_history = session.get("chat_history", [])

        # Xử lý RAG
        answer_raw, raw_docs = ask_question(chain, question, context_history)
        answer_html = markdown.markdown(answer_raw, extensions=['tables', 'fenced_code'])
        safe_sources = simplify_sources(raw_docs)
        
        # Tạo object tin nhắn
        user_msg = {"role": "user", "content": question, "timestamp": datetime.now()}
        bot_msg = {
            "role": "assistant", 
            "content": answer_html, 
            "sources": safe_sources, 
            "model": model, 
            "timestamp": datetime.now()
        }

        # LƯU TRỮ
        if session.get("user_id") and db is not None:
            user_id = session["user_id"]
            chat_id = session.get("current_chat_id")

            if chat_id:
                # Cập nhật chat hiện tại
                conversations_col.update_one(
                    {"_id": ObjectId(chat_id)},
                    {
                        "$push": {"messages": {"$each": [user_msg, bot_msg]}},
                        "$set": {"updated_at": datetime.now()}
                    }
                )
            else:
                # Tạo hội thoại mới
                new_chat = {
                    "user_id": user_id,
                    "title": question[:50] + "..." if len(question) > 50 else question,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                    "messages": [user_msg, bot_msg]
                }
                res = conversations_col.insert_one(new_chat)
                session["current_chat_id"] = str(res.inserted_id)
        else:
            # Lưu Session cho khách
            hist = session.get("chat_history", [])
            hist.append(user_msg)
            hist.append(bot_msg)
            session["chat_history"] = hist
            session.modified = True

        return jsonify({
            "answer": answer_html,
            "sources": safe_sources,
            "model": model
        })

    except Exception as e:
        print(f"[ERROR]: {str(e)}", flush=True)
        return jsonify({"error": str(e)}), 500

@app.route("/new_chat", methods=["POST"])
def new_chat():
    """Tạo phiên chat mới"""
    if session.get("user_id"):
        session["current_chat_id"] = None
    else:
        session.pop("chat_history", None)
    return jsonify({"status": "success"})

@app.route("/delete_chat", methods=["POST"])
def delete_chat():
    """Xóa hội thoại hiện tại"""
    if session.get("user_id") and session.get("current_chat_id") and db is not None:
        conversations_col.delete_one({"_id": ObjectId(session["current_chat_id"])})
        session["current_chat_id"] = None
    else:
        session.pop("chat_history", None)
    return jsonify({"status": "success"})

@app.route("/feedback", methods=["POST"])
def feedback():
    """Lưu phản hồi"""
    try:
        data = request.get_json()
        feedback_doc = {
            "type": data.get("type"),
            "question": data.get("question"),
            "answer": data.get("answer"),
            "model": data.get("model", "unknown"),
            "user_id": session.get("user_id", "guest"),
            "timestamp": datetime.now(timezone.utc)
        }
        if db is not None:
            result = feedback_col.insert_one(feedback_doc)
            return jsonify({"status": "success", "id": str(result.inserted_id)})
        else:
            with open("feedback_logs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            return jsonify({"status": "success", "storage": "local"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clear", methods=["POST"])
def clear():
    session.pop("chat_history", None)
    return jsonify({"status": "success"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)