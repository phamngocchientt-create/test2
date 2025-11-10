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

# --- TÌM KIẾM NGỮ NGHĨA VỚI FAISS ---
@st.cache_resource
def build_semantic_index(knowledge_texts):
    if not knowledge_texts:
        return None
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    chunks, meta = [], []
    for item in knowledge_texts:
        for para in item["content"].split("\n"):
            para = para.strip()
            if len(para) > 60:
                chunks.append(para)
                meta.append(item["filename"])

    emb = model.encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.array(emb, dtype=np.float32))

    return {"index": index, "model": model, "chunks": chunks, "meta": meta}

semantic_index = build_semantic_index(knowledge_texts)

def search_knowledge_semantic(query, top_k=3):
    if not semantic_index:
        return None
    model = semantic_index["model"]
    index = semantic_index["index"]
    chunks = semantic_index["chunks"]
    meta = semantic_index["meta"]

    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb, dtype=np.float32), top_k)

    results = []
    for idx, score in zip(I[0], D[0]):
        if score < 0.25:
            continue
        results.append(f"📘 [Tài liệu: {meta[idx]}]\n{chunks[idx]}")
    return "\n\n---\n".join(results) if results else None

# --- HỆ THỐNG CHAT ---
if "chat_session" not in st.session_state:
    system_instruction = r"""
Bạn là "Gia Sư AI Hóa học THCS" — chuyên nghiệp, thân thiện, và kiên nhẫn.
✅ ƯU TIÊN TUYỆT ĐỐI: Nếu có tài liệu liên quan trong '📚 KIẾN THỨC CẦN THAM KHẢO', phải dùng nó trước và trích dẫn nguồn (VD: Theo [Tên file]).
Chỉ khi không có kiến thức trong tài liệu thì mới được phép dùng kiến thức nền của bạn.
Câu trả lời phải bằng tiếng Việt, rõ ràng, có giải thích từng bước, trình bày công thức bằng LaTeX.
"""
    config = types.GenerateContentConfig(system_instruction=system_instruction)
    st.session_state.chat_session = client.chats.create(model="gemini-2.5-pro", config=config)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- GIAO DIỆN NGƯỜI DÙNG ---
uploaded_file = st.file_uploader("📷 Tải ảnh bài tập (JPG/PNG)", type=["jpg", "jpeg", "png"])
user_question = st.chat_input("✏️ Nhập câu hỏi Hóa học...")

if user_question:
    kb_context = search_knowledge_semantic(user_question)
    contents = []
    if uploaded_file:
        img_part = types.Part.from_bytes(data=uploaded_file.read(), mime_type=uploaded_file.type)
        contents.append(img_part)

    full_prompt = (
        f"📚 KIẾN THỨC CẦN THAM KHẢO:\n{kb_context}\n\n---\n\n{user_question}"
        if kb_context else user_question
    )
    contents.append(full_prompt)

    with st.chat_message("Học sinh"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "Học sinh", "content": user_question})

    with st.spinner("⏳ Gia sư đang trả lời..."):
        try:
            response = st.session_state.chat_session.send_message(contents)
            reply = response.text
        except Exception as e:
            reply = f"⚠️ Lỗi xử lý: {e}"

    with st.chat_message("Gia Sư"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "Gia Sư", "content": reply})
    st.rerun()

# --- QUẢN TRỊ ---
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
