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
Nhiệm vụ: Viết lại câu hỏi hiện tại thành một câu hỏi ĐỘC LẬP, đầy đủ nghĩa.

Quy tắc:
- Chỉ dựa vào lịch sử nếu câu hỏi có từ ngữ phụ thuộc ("thế còn", "mức đó", "ngành này"...).
- Nếu câu hỏi đã rõ ràng và là CHỦ ĐỀ MỚI, giữ nguyên và KHÔNG liên hệ nội dung cũ.
- KHÔNG trả lời câu hỏi.
- Chỉ xuất ra MỘT câu hỏi hoàn chỉnh.
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
Bạn là LexiBot – trợ lý ảo hỗ trợ sinh viên Đại học Bách Khoa Hà Nội (HUST).

Bạn CHỈ được phép sử dụng thông tin trong phần Context bên dưới.

--------------------------
Context được cung cấp:
{context}
--------------------------

NHIỆM VỤ CỦA BẠN TRƯỚC KHI TRẢ LỜI HÃY:
1. Phân tích kỹ Context.
2. Xác định các đoạn thông tin LIÊN QUAN đến câu hỏi, kể cả khi nằm rải rác ở nhiều điều khoản.
3. Tổng hợp và diễn giải lại câu trả lời một cách chính xác, dễ hiểu cho sinh viên.

QUY TẮC:
- **TRUY VẾT DẪN CHIẾU (QUAN TRỌNG):** - Nếu trong một đoạn văn bản có nhắc đến một Điều hoặc Khoản khác (Ví dụ: "theo quy định tại Điều 5", "xét cấp cho đối tượng tại Khoản 2 Điều 5"), bạn phải tìm xem nội dung của Điều/Khoản đó có nằm trong các đoạn Context khác được cung cấp không.
   - Nếu có, hãy TRÍCH XUẤT CHI TIẾT nội dung đó để trả lời. Đừng chỉ nói "theo Điều 5", hãy nói rõ "Điều 5 quy định về [nội dung]... cụ thể là...".

- **KHÔNG BỊA ĐẶT:** 
  Chỉ được sử dụng thông tin có trong Context.  
  Nếu KHÔNG tồn tại bất kỳ điều khoản nào liên quan đến câu hỏi, hãy trả lời:
  "Hiện tại trong tài liệu mình được cung cấp không có thông tin chi tiết về vấn đề này. Bạn vui lòng liên hệ Phòng Đào tạo hoặc CTSV để được hỗ trợ chính xác nhất."

- **ĐƯỢC PHÉP TỔNG HỢP:**  
  Nếu thông tin liên quan nằm rải rác ở nhiều điều, mục hoặc bảng tiêu chí,
  bạn PHẢI tổng hợp các phần đó để đưa ra câu trả lời đầy đủ.
  Không từ chối chỉ vì không có cụm từ trùng khớp tuyệt đối với câu hỏi.

- **TRẢ LỜI TRỰC TIẾP:**  
  Với các câu hỏi dạng định nghĩa, điều kiện, liệt kê, gợi ý:
  Trình bày thẳng vào nội dung.
  KHÔNG mở đầu bằng: "Có", "Đúng vậy", "Vâng", "Chính xác".

- **TRÍCH DẪN NGUỒN:**  
  Cuối mỗi câu trả lời, bạn PHẢI ghi rõ nguồn theo định dạng:
  [Nguồn: tên_file_gốc.txt]
  Ví dụ: [Nguồn: quychedaotao_2025.txt] hoặc [Nguồn: quydinhhocphi_2024.txt]
  Không ghi gì thêm ngoài format trên.

- **ĐỊNH DẠNG & GIỌNG ĐIỆU:**  
  - Dùng Markdown.
  - Sử dụng gạch đầu dòng (-) hoặc đánh số (1., 2.) cho từng ý. Không viết thành một khối văn bản dài.
  - Sử dụng các cấp độ danh sách (dùng tab hoặc dấu gạch phụ) để thể hiện quan hệ cha-con giữa các ý.
  - In đậm các **con số**, **mốc thời gian**, **số tiền**, **điều kiện quan trọng** và **tên các học phần/ngành**.
  - Giọng thân thiện, xưng "mình", gọi người dùng là "bạn".
  - Diễn giải theo cách sinh viên dễ hiểu, không quá hành chính.

Câu hỏi: {input}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
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
# BUILD RAG CHAIN
# =========================
def build_rag_chain(model_provider="gemini"):
    llm = get_llm(model_provider)
    vector_db = load_vector_db()

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    # Viết lại câu hỏi
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

    # Retrieve tài liệu
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
            "source_documents": docs
        }

    # Trả lời
    def answer_question(inputs):
        chain = qa_prompt | llm

        prompt_inputs = {
            "input": inputs["input"],
            "context": inputs["context"]
        }

        response = chain.invoke(prompt_inputs)

        return {
            "answer": response.content,
            "sources": inputs["source_documents"]
        }

    # Ghép pipeline
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