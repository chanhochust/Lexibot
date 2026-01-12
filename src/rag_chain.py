import os
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

from src.models import get_embedding_model, get_llm

DB_PATH = "./chroma_db"

# =========================
# PROMPT VIẾT LẠI CÂU HỎI
# =========================
contextualize_q_system_prompt = """
Nhiệm vụ: Viết lại câu hỏi của người dùng thành một câu hoàn chỉnh, rõ nghĩa dựa trên lịch sử trò chuyện.

Quy tắc:
- Nếu câu hỏi phụ thuộc vào câu trước, hãy viết lại đầy đủ.
- Làm rõ các thực thể bị ẩn (nó, cái đó, mức đó,...) dựa vào tin nhắn trước.
- Nếu câu hỏi đã rõ ràng, giữ nguyên.
- KHÔNG trả lời câu hỏi ở bước này.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# =========================
# PROMPT TRẢ LỜI (QA)
# =========================
qa_system_prompt = """
Bạn là LexiBot – trợ lý ảo thông minh hỗ trợ sinh viên Đại học Bách Khoa Hà Nội (HUST).

Dưới đây là thông tin trích xuất từ quy định (Context):
--------------------------
{context}
--------------------------

NHIỆM VỤ:
Trả lời câu hỏi của sinh viên dựa trên Context trên.

QUY TẮC:
1.  **Trả lời trực tiếp:** Nếu thấy thông tin liên quan trong Context, hãy tổng hợp và trả lời ngay. Đừng nói "Tôi không thấy" nếu sau đó bạn vẫn đưa ra được dữ liệu.
2.  **Trích dẫn:** Nêu rõ tên văn bản hoặc Điều/Mục (Ví dụ: "Theo Điều 5 Quy chế đào tạo...").
3.  **Trung thực:** Chỉ khi Context hoàn toàn không có thông tin liên quan, hãy nói: "Xin lỗi, hiện tại tài liệu mình có chưa đề cập đến vấn đề này."
4.  **Trình bày:** Sử dụng Markdown (gạch đầu dòng, in đậm các ý chính).
5.  **Giọng điệu:** Thân thiện, hỗ trợ, xưng "mình" và gọi "bạn".

Câu hỏi: {input}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# =========================
# LOAD VECTOR DB
# =========================
def load_vector_db():
    if not os.path.exists(DB_PATH):
        raise RuntimeError("Chưa có chroma_db, hãy chạy create_db.py trước")

    embedding = get_embedding_model()
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding
    )

# =========================
# BUILD RAG CHAIN (LCEL)
# =========================
def build_rag_chain():
    llm = get_llm()
    vector_db = load_vector_db()

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    # --- STEP 1: Viết lại câu hỏi ---
    def rewrite_question(inputs):
        history = inputs.get("chat_history", [])

        if not history:
            return {
                "question": inputs["question"],
                "chat_history": []
            }

        chain = contextualize_q_prompt | llm
        rewritten = chain.invoke({
            "input": inputs["question"],
            "chat_history": history
        })

        return {
            "question": rewritten.content,
            "chat_history": history
        }

    # --- STEP 2: Retrieve tài liệu ---
    def retrieve_docs(inputs):
        question = inputs["question"]
        docs = retriever.invoke(question)

        context_parts = []
        for d in docs:
            file_name = os.path.basename(d.metadata.get('source', 'Tai_lieu'))
            section = d.metadata.get('section', 'Thông tin chung')
            part = f"[Nguồn: {file_name} | {section}]\nNội dung: {d.page_content}"
            context_parts.append(part)

        context_text = "\n\n".join(context_parts)

        return {
            "input": question,
            "context": context_text,
            "chat_history": inputs["chat_history"],
            "source_documents": docs
        }

    # --- STEP 3: Trả lời ---
    def answer_question(inputs):
        chain = qa_prompt | llm

        prompt_inputs = {
            "input": inputs["input"],
            "context": inputs["context"],
            "chat_history": inputs["chat_history"]
        }

        response = chain.invoke(prompt_inputs)

        return {
            "answer": response.content,
            "sources": inputs["source_documents"]
        }

    # --- GHÉP PIPELINE ---
    rag_chain = (
        RunnableLambda(rewrite_question)
        | RunnableLambda(retrieve_docs)
        | RunnableLambda(answer_question)
    )

    return rag_chain

# =========================
# ASK (STATELESS)
# =========================
def ask_question(chain, question: str, chat_history: list = None):
    if chain is None:
        return "Hệ thống chưa sẵn sàng.", []

    if chat_history is None:
        chat_history = []

    processed_history = []
    for msg in chat_history:
        if msg["role"] == "user":
            processed_history.append(HumanMessage(content=msg["content"]))
        else:
            processed_history.append(AIMessage(content=msg["content"]))

    response = chain.invoke({
        "question": question,
        "chat_history": processed_history
    })

    return response["answer"], response.get("sources", [])
