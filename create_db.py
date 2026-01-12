import os
import shutil
import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.models import get_embedding_model


DATA_PATH = "./data"
DB_PATH = "./chroma_db"
MAX_CHUNK_SIZE = 1500  # Kích thước tối đa của 1 chunk (ký tự)

def create_vector_db():
    print("BẮT ĐẦU TẠO VECTOR DATABASE")

    # Dọn dẹp DB cũ
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"Đã xóa database cũ tại {DB_PATH}")

    # Load tài liệu
    if not os.path.exists(DATA_PATH):
        print(f"Thư mục {DATA_PATH} không tồn tại!")
        return

    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    raw_documents = loader.load()
    print(f"Đã tải {len(raw_documents)} file tài liệu")

    # Xử lý & Chunking
    all_chunks = []
    
    for doc in raw_documents:
        file_name = os.path.basename(doc.metadata.get("source", ""))
        content = doc.page_content
        
        # Phân loại tài liệu để áp dụng chiến thuật cắt
        if is_legal_document(file_name):
            print(f"Xử lý Quy chế: {file_name}")
            chunks = split_legal_document(content, doc.metadata)
        else:
            print(f"Xử lý Sổ tay/Markdown: {file_name}")
            chunks = split_markdown_document(content, doc.metadata)
            
        all_chunks.extend(chunks)

    print(f"Tổng số chunk tạo ra: {len(all_chunks)}")
    
    # In 3 chunk đầu
    print_debug_chunks(all_chunks)

    # Lưu vào ChromaDB
    print("\nĐang mã hóa (Embedding) và lưu vào DB...")
    embedding_model = get_embedding_model()
    vector_db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    vector_db.persist()
    print(f"HOÀN TẤT! Database sẵn sàng tại: {DB_PATH}")



def is_legal_document(filename):
    """Nhận diện file quy chế dựa trên tên file"""
    keywords = ["quyche", "quydinh", "quyetdinh", "luat", "daotao", "hocphi"]
    return any(k in filename.lower() for k in keywords)

def split_legal_document(text, metadata):
    """
    Chiến thuật cho Quy chế:
    1. Tách theo Chương (để lấy ngữ cảnh lớn).
    2. Trong Chương, tách theo Điều.
    3. Trong Điều, nếu dài quá thì tách theo Khoản (1., 2.) hoặc ý nhỏ.
    QUAN TRỌNG: Luôn gắn 'Điều X...' vào đầu mỗi chunk con.
    """
    chunks = []
    
    # Tách các Chương
    # Regex: Tìm chuỗi "CHƯƠNG [Số La Mã]"
    chapter_splits = re.split(r"(^CHƯƠNG\s+[IVXLCDM]+.*$)", text, flags=re.MULTILINE)
    
    current_chapter = "Quy định chung"
    
    for i in range(1, len(chapter_splits), 2):
        if i+1 < len(chapter_splits):
            header = chapter_splits[i].strip() # Tên chương
            body = chapter_splits[i+1]         # Nội dung chương
            
            # Trong mỗi chương, tách các Điều
            # Regex: Tìm "Điều [Số]."
            article_splits = re.split(r"(^Điều\s+\d+[\.:]?\s+.*$)", body, flags=re.MULTILINE)
            
            current_article_header = ""
            
            # Xử lý phần dẫn nhập của chương (nếu có)
            if article_splits[0].strip():
                 chunks.append(create_doc(
                     text=article_splits[0], 
                     meta=metadata, 
                     context=f"{header}"
                 ))

            for k in range(1, len(article_splits), 2):
                if k+1 < len(article_splits):
                    art_header = article_splits[k].strip() # VD: "Điều 5. Điểm học phần"
                    art_body = article_splits[k+1]         # Nội dung điều
                    
                    full_context_header = f"{header} > {art_header}"
                    
                    # Kiểm tra độ dài Điều
                    full_text = f"{art_header}\n{art_body}"
                    
                    if len(full_text) < MAX_CHUNK_SIZE:
                        # Nếu ngắn, giữ nguyên cả điều
                        chunks.append(create_doc(art_body, metadata, full_context_header))
                    else:
                        # Nếu dài, cắt nhỏ nhưng LUÔN KÈM TIÊU ĐỀ ĐIỀU
                        sub_chunks = recursive_split(art_body, chunk_size=1000)
                        for sub in sub_chunks:
                            # Context Injection: Gắn tiêu đề vào nội dung để AI hiểu
                            chunks.append(create_doc(sub, metadata, full_context_header))
                            
    # Xử lý trường hợp văn bản không có Chương, chỉ có Điều
    if not chunks: 
        # Fallback: Tách thẳng theo Điều
        article_splits = re.split(r"(^Điều\s+\d+[\.:]?\s+.*$)", text, flags=re.MULTILINE)
        for k in range(1, len(article_splits), 2):
            header = article_splits[k].strip()
            body = article_splits[k+1]
            chunks.append(create_doc(body, metadata, header))
            
    return chunks

def split_markdown_document(text, metadata):
    """
    Chiến thuật cho Sổ tay (Markdown):
    Cắt theo cấp độ Header: # -> ## -> ###
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_docs = markdown_splitter.split_text(text)
    
    final_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_SIZE, chunk_overlap=200)
    
    for doc in md_docs:
        # Tạo context string từ metadata header
        context_parts = []
        if "Header 1" in doc.metadata: context_parts.append(doc.metadata["Header 1"])
        if "Header 2" in doc.metadata: context_parts.append(doc.metadata["Header 2"])
        if "Header 3" in doc.metadata: context_parts.append(doc.metadata["Header 3"])
        
        context_str = " > ".join(context_parts)
        
        # Nếu chunk quá dài, cắt nhỏ thành các phần
        if len(doc.page_content) > MAX_CHUNK_SIZE:
            splits = text_splitter.split_text(doc.page_content)
            for s in splits:
                final_chunks.append(create_doc(s, metadata, context_str))
        else:
            final_chunks.append(create_doc(doc.page_content, metadata, context_str))
            
    return final_chunks

def recursive_split(text, chunk_size):
    """Hàm cắt nhỏ bổ trợ"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def create_doc(text, meta, context):
    """
    Tạo Document chuẩn hóa.
    QUAN TRỌNG: Gộp Context vào page_content để Embedding hiểu ngữ cảnh.
    """
    # Làm sạch text
    text = re.sub(r'\n+', '\n', text).strip()
    
    # Nội dung thực tế đưa vào Vector DB = [Tiêu đề] + [Nội dung]
    # Ví dụ: "Điều 5. Học phí... [Nội dung chi tiết]"
    content_with_context = f"[{context}]\n{text}"
    
    new_meta = meta.copy()
    new_meta["section"] = context # Lưu tiêu đề để hiển thị nguồn sau này
    
    return Document(page_content=content_with_context, metadata=new_meta)

def print_debug_chunks(chunks):
    print("\n🔍 --- In 3 chunk đầu tiên ---")
    for i, c in enumerate(chunks[:3]):
        print(f"Chunk {i+1}:")
        print(f"   📂 File: {c.metadata.get('source')}")
        print(f"   🏷️  Section: {c.metadata.get('section')}")
        print(f"   📝 Content (Preview): {c.page_content[:150]}...")
        print("-" * 50)

if __name__ == "__main__":
    create_vector_db()