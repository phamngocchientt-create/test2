import streamlit as st
import os
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# --- CẤU HÌNH ỨNG DỤNG ---
st.set_page_config(page_title="Gia Sư Hóa Học THCS", page_icon="🧪")
st.title("🧪 Gia Sư Hóa Học THCS")

# --- KHỞI TẠO GEMINI CLIENT ---
@st.cache_resource
def get_gemini_client():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.session_state.admin_password = st.secrets.get("ADMIN_PASSWORD", "211191")
        return genai.Client(api_key=api_key)
    except (AttributeError, KeyError):
        try:
            st.session_state.admin_password = os.getenv("ADMIN_PASSWORD", "211191")
            return genai.Client()
        except Exception:
            st.error("❌ Không tìm thấy GEMINI API Key. Vui lòng thêm vào Streamlit Secrets.")
            st.stop()

client = get_gemini_client()

# --- QUẢN LÝ TÀI LIỆU ---
knowledge_path = "knowledge_base"
os.makedirs(knowledge_path, exist_ok=True)

def extract_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".pdf":
            reader = PdfReader(filepath)
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
            return text
        elif ext == ".docx":
            doc = Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return None
    return None

@st.cache_resource
def load_knowledge_base():
    texts = []
    for fn in os.listdir(knowledge_path):
        path = os.path.join(knowledge_path, fn)
        if os.path.splitext(fn)[1].lower() in {".txt", ".pdf", ".docx"}:
            content = extract_text_from_file(path)
            if content and content.strip():
                texts.append({"filename": fn, "content": content})
    return texts

knowledge_texts = load_knowledge_base()

# --- TÌM KIẾM NGỮ NGHĨA (FAISS) ---
@st.cache_resource
def build_semantic_index(knowledge_texts):
    if not knowledge_texts:
        return None
    # Model được tinh chỉnh cho đa ngôn ngữ, hỗ trợ tiếng Việt tốt
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") 

    chunks, meta = [], []
    for item in knowledge_texts:
        # Tách tài liệu thành các đoạn văn ngắn hơn
        for para in item["content"].split("\n"):
            para = para.strip()
            # Bỏ qua các đoạn quá ngắn
            if len(para) > 60: 
                chunks.append(para)
                meta.append(item["filename"])

    emb = model.encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.array(emb, dtype=np.float32))

    return {"index": index, "model": model, "chunks": chunks, "meta": meta}

semantic_index = build_semantic_index(knowledge_texts)

def search_knowledge_semantic(query, top_k=5): # Giảm top_k xuống 5 để Context cô đọng hơn
    if not semantic_index:
        return None
    model = semantic_index["model"]
    index = semantic_index["index"]
    chunks = semantic_index["chunks"]
    meta = semantic_index["meta"]

    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb, dtype=np.float32), top_k)

    results = []
    # Chỉ lấy các kết quả có điểm tương đồng (score) cao
    for idx, score in zip(I[0], D[0]): 
        # Ngưỡng (Threshold) đã điều chỉnh để yêu cầu sự liên quan cao hơn
        if score > 0.65: 
            results.append(f"📘 [Tài liệu: {meta[idx]}]\n{chunks[idx]}")
    return "\n\n---\n".join(results) if results else None

# --- HỆ THỐNG CHAT ---
if "chat_session" not in st.session_state:
    # 📌 ĐÃ SỬA: Nâng cấp System Instruction
    system_instruction = r"""
BẠN LÀ AI: Bạn là "Gia Sư AI Hóa học THCS" – chuyên nghiệp, thân thiện, và kiên nhẫn.
Mục tiêu: Hướng dẫn học sinh hiểu và giải bài tập Hóa học.

**QUY TẮC CHƯƠNG TRÌNH & THUẬT NGỮ:**
1. **Tuân thủ Tuyệt đối:** PHẢI tuân thủ **Chương trình Giáo dục Phổ thông 2018**. Tránh kiến thức cũ trừ khi được hỏi cụ thể.
2. **Thuật ngữ thống nhất:** Sử dụng thuật ngữ Hóa học theo chương trình mới (Ví dụ: Acid, Base, Oxide, Sodium, Potassium) thay vì tiếng Việt (axit, bazơ, oxit, natri, kali).
3. **Thể tích mol:** Luôn sử dụng điều kiện chuẩn ($\text{25}^{\circ}\text{C}$ và $1\ \text{bar}$), thể tích mol là $24,79\ \text{L}/\text{mol}$, trừ khi đề bài ghi rõ ĐKTC ($0^{\circ}\text{C}$ và $1\ \text{atm}$).

1. **QUY TẮC BẮT BUỘC SỬ DỤNG VÀ TRÍCH DẪN KIẾN THỨC (CONTEXT)**
    - KHU VỰC CONTEXT (Nguồn thông tin DUY NHẤT để trích dẫn) được xác định bởi thẻ **<KB_START>** và **<KB_END>**.
    - **ƯU TIÊN TUYỆT ĐỐI:** NẾU có Context liên quan (<KB_START>...</KB_END>), bạn PHẢI dựa hoàn toàn vào đó để trả lời.
    - **CÁCH TRÍCH DẪN BẮT BUỘC:** Bạn PHẢI trích dẫn nguồn ngay sau khi sử dụng thông tin đó (Ví dụ: Theo [Tên file]).
    - **HÌNH PHẠT:** KHÔNG được trích dẫn bất kỳ nguồn nào KHÔNG nằm trong khu vực <KB_START>...</KB_END>. Nếu trích dẫn sai hoặc bỏ qua Context liên quan, câu trả lời bị coi là không chuyên biệt.
    - **FALLBACK:** Chỉ khi Context không có, mới được dùng kiến thức nền tảng và **KHÔNG TRÍCH DẪN NGUỒN**.

2. **ĐỊNH DẠNG TRẢ LỜI:**
    - Trả lời bằng tiếng Việt, chi tiết từng bước.
    - **LaTeX:** Mọi công thức, phương trình, đơn vị và ký hiệu PHẢI được bọc trong cú pháp $\text{\LaTeX}$ (dùng '$' hoặc '$$').
"""
    config = types.GenerateContentConfig(system_instruction=system_instruction)
    st.session_state.chat_session = client.chats.create(model="gemini-2.5-flash", config=config)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- GIAO DIỆN ---
uploaded_file = st.file_uploader("📷 Tải ảnh bài tập (JPG/PNG)", type=["jpg", "jpeg", "png"])
user_question = st.chat_input("✏️ Nhập câu hỏi Hóa học...")

if user_question:
    # Bước 1: Tìm kiếm Context
    kb_context = search_knowledge_semantic(user_question)
    
    contents = []

    if uploaded_file:
        img_part = types.Part.from_bytes(data=uploaded_file.read(), mime_type=uploaded_file.type)
        contents.append(img_part)

    # Bước 2: Xây dựng Prompt (Phân biệt Có KB và Không có KB)
    if kb_context:
        # Trường hợp 1: CÓ Context (Áp dụng thẻ KB_START/KB_END)
        full_prompt = f"""
<KB_START>
📚 KIẾN THỨC CẦN THAM KHẢO:
{kb_context}
<KB_END>

--- HỎI ĐÁP ---
Câu hỏi của học sinh:
{user_question}
"""
    else:
        # Trường hợp 2: KHÔNG CÓ Context
        full_prompt = f"""
Không có tài liệu tham khảo liên quan được tìm thấy.
Hãy trả lời dựa trên kiến thức nền tảng của bạn (theo Chương trình GDPT 2018).

Câu hỏi:
{user_question}
"""

    contents.append(full_prompt)

    # Bước 3: Gửi và Hiển thị
    with st.chat_message("Học sinh"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "Học sinh", "content": user_question})

    with st.spinner("⏳ Gia sư đang trả lời..."):
        try:
            response = st.session_state.chat_session.send_message(contents)
            reply = response.text
        except Exception as e:
            reply = f"⚠️ Lỗi xử lý API Gemini: {type(e).__name__}: {e}. Vui lòng thử lại hoặc hỏi câu khác."

    with st.chat_message("Gia Sư"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "Gia Sư", "content": reply})
    st.rerun()

# --- KHU VỰC QUẢN TRỊ ---
with st.sidebar:
    st.header("🔐 Khu vực quản trị")
    pwd = st.text_input("Nhập mật khẩu admin:", type="password")
    if "admin_password" in st.session_state and pwd == st.session_state.admin_password:
        st.success("✅ Đăng nhập thành công!")
        st.info(f"Tổng số tài liệu: **{len(knowledge_texts)}**")
        st.markdown("📂 Thư mục: `/knowledge_base` (chứa tài liệu .pdf, .docx, .txt)")
        st.markdown("🔁 Sau khi thêm tài liệu, **restart lại app** để cập nhật.")
    elif pwd:
        st.error("❌ Sai mật khẩu!")
