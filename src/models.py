import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

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

def get_llm(model_provider: str = "gemini"):
    # Gemini
    if model_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY_2")
        if not api_key:
            raise ValueError("Không tìm thấy GOOGLE_API_KEY")

        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-09-2025",
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=2048,
        )
    # Dùng Groq 
    elif model_provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Không tìm thấy GROQ_API_KEY!\n")
    
        return ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            model="llama-3.3-70b-versatile", # Hoặc "llama-3.1-8b-instant"
            temperature=0.3,
            max_tokens=2048,
            timeout=60,  # Tăng timeout
        )
    else:
        raise ValueError(f"Model provider không hợp lệ: {model_provider}")