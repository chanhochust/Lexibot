import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

_embedding_model_instance = None

def get_embedding_model():
    global _embedding_model_instance
    if _embedding_model_instance is not None:
        return _embedding_model_instance
    
    print("Đang tải model Embedding...")
    _embedding_model_instance = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", # Hoặc "intfloat/multilingual-e5-base"
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Embedding model ready!")
    return _embedding_model_instance

def get_llm():
     # Dùng Groq (FREE)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Không tìm thấy GROQ_API_KEY!\n")
    
    model_name = "llama-3.3-70b-versatile" # Hoặc "gemma2-9b-it" "qwen-2.5-72b-versatile" "mixtral-8x7b-32768"
        
    return ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        model=model_name,
        temperature=0.3,
        max_tokens=2048,
        timeout=60,  # Tăng timeout
    )