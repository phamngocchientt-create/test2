import streamlit as st
import os
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import time # Cần thiết cho cơ chế thử lại API

# --- CẤU HÌNH ỨNG DỤNG ---
st.set_page_config(page_title="Gia Sư Hóa Học THCS", page_icon="🧪")
st.title("🧪 Gia Sư Hóa Học THCS")

# --- KHỞI TẠO TRẠNG THÁI (SESSION STATE) ---
if 'file_key' not in st.session_state:
    st.session_state['file_key'] = 0
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if "messages" not in st.session_state:
    st.session_state.messages = []
    
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

# --- QUẢN LÝ TÀI LIỆU (Giữ nguyên) ---
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

def search_knowledge_semantic(query, top_k=5):
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
        if score > 0.65:
            results.append(f"📘 [Tài liệu: {meta[idx]}]\n{chunks[idx]}")
    return "\n\n---\n".join(results) if results else None

# --- HỆ THỐNG CHAT ---
if "chat_session" not in st.session_state:
   system_instruction = r"""
BẠN LÀ AI: Bạn là "Gia S Sư AI Hóa học THCS" – chuyên nghiệp, thân thiện, và kiên nhẫn.
Mục tiêu: Hướng dẫn học sinh hiểu và giải bài tập Hóa học.

**QUY TẮC CHƯƠNG TRÌNH & THUẬT NGỮ:**
1. **Tuân thủ Tuyệt đối:** PHẢI tuân thủ **Chương trình Giáo dục Phổ thông 2018**. Tránh kiến thức cũ trừ khi được hỏi cụ thể.
2. **Thuật ngữ thống nhất:** Sử dụng thuật ngữ Hóa học theo chương trình mới (Ví dụ: Acid, Base, Oxide, Sodium, Potassium) thay vì tiếng Việt (axit, bazơ, oxit, natri, kali).
3. **Ký hiệu Trạng thái (MỚI):** **TUYỆT ĐỐI KHÔNG** sử dụng các ký hiệu trạng thái trong ngoặc đơn như (r), (l), (dd), (k). Thay vào đó, PHẢI sử dụng mũi tên đi lên ($\uparrow$) cho chất khí và mũi tên đi xuống ($\downarrow$) cho chất kết tủa.
4. **Thể tích mol:** Luôn sử dụng điều kiện chuẩn ($\text{25}^{\circ}\text{C}$ và $1\ \text{bar}$), thể tích mol là $24,79\ \text{L}/\text{mol}$, trừ khi đề bài ghi rõ ĐKTC ($0^{\circ}\text{C}$ và $1\ \text{atm}$).

1. **QUY TẮC BẮT BUỘC SỬ DỤNG VÀ TRÍCH DẪN KIẾN THỨC (CONTEXT)**
    - KHU VỰC CONTEXT (Nguồn thông tin DUY NHẤT để trích dẫn) được xác định bởi thẻ **<KB_START>** và **<KB_END>**.
    - **ƯU TIÊN TUYỆT ĐỐI (SIẾT CHẶT):** NẾU có Context liên quan (<KB_START>...</KB_END>), bạn PHẢI dựa **HOÀN TOÀN** vào đó để trả lời.
    - **CÁCH TRÍCH DẪN BẮT BUỘC:** Bạn PHẢI trích dẫn nguồn ngay sau khi sử dụng thông tin đó (Ví dụ: Theo [Tên file]).
    - **HÌNH PHẠT:** KHÔNG được trích dẫn bất kỳ nguồn nào KHÔNG nằm trong khu vực <KB_START>...</KB_END>. Nếu trích dẫn sai hoặc bỏ qua Context liên quan, câu trả lời bị coi là không chuyên biệt.
    - **FALLBACK BỊ CẤM (SIẾT CHẶT):** Nếu Context được cung cấp, bạn **TUYỆT ĐỐI KHÔNG** được sử dụng kiến thức nền tảng của mình (Kiến thức của Google). **Chỉ khi HOÀN TOÀN KHÔNG CÓ Context liên quan**, bạn mới được phép dùng kiến thức nền tảng và **KHÔNG TRÍCH DẪN NGUỒN**.

2. **ĐỊNH DẠNG TRẢ LỜI:**
    - Trả lời bằng tiếng Việt, chi tiết từng bước.
    - **QUY TẮC PHÂN BIỆT RÕ RÀNG (CHỈNH SỬA VÀ TĂNG CƯỜNG):**
        - **LỜI HƯỚNG DẪN & GIẢI THÍCH:** Bất kỳ câu nào mang tính chất **trò chuyện, hướng dẫn, gợi ý, hoặc giải thích ý nghĩa của bước làm** (như Gia Sư đang nói chuyện với học sinh) **PHẢI được đặt trong ngoặc kép ("...")**. 
          Ví dụ: "Muốn tính được khối lượng của $\text{Fe}$ đầu tiên ta sẽ phải tìm số mol của nó."
        - **LỜI GIẢI, CÔNG THỨC & KẾT QUẢ:** Các bước **tính toán thực tế, áp dụng công thức, các phép tính, và đáp án cuối cùng** **PHẢI được tô đậm (dùng **...)** và KHÔNG ĐƯỢC đặt trong ngoặc kép.
          Ví dụ: **Số mol của $\text{Fe}$ thu được là:**; **$n_{\text{Fe}} = 0,2\ \text{mol}$**
    - **LaTeX:** Mọi công thức, phương trình, đơn vị và ký hiệu PHẢI được bọc trong cú pháp \LaTeX (dùng '$' hoặc '$$').
"""
st.session_state.chat_session = client.chats.create(model="gemini-2.5-flash", config=config)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- GIAO DIỆN VÀ XỬ LÝ INPUT (ĐÃ SỬA LỖI LẶP VÀ HỎI LẠI UX) ---
uploaded_file = st.file_uploader("📷 Tải ảnh bài tập (JPG/PNG)", 
                                 type=["jpg", "jpeg", "png"],
                                 key=st.session_state['file_key'])
user_question = st.chat_input("✏️ Nhập câu hỏi Hóa học...")

# 1. Logic xử lý ảnh: Lưu ảnh và hỏi lại (Ngăn chatbot tự ý trả lời)
if uploaded_file and not user_question and st.session_state.uploaded_image is None:
    st.session_state.uploaded_image = {
        "bytes": uploaded_file.read(),
        "type": uploaded_file.type
    }
    st.session_state['file_key'] += 1
    
    st.session_state.messages.append({"role": "Học sinh", "content": "[Ảnh bài tập đã tải lên]"})
    st.session_state.messages.append({"role": "Gia Sư",
                                      "content": "Bạn đã tải ảnh bài tập lên thành công. **Bạn muốn tôi làm gì với bài tập trên?**"})
    st.rerun()

# 2. Logic xử lý yêu cầu khi có văn bản (có thể kèm ảnh đang chờ)
if user_question:

    image_bytes = None
    image_type = None

    # Lấy ảnh đang chờ, nếu có (từ lần upload trước)
    if st.session_state.uploaded_image is not None:
        image_bytes = st.session_state.uploaded_image["bytes"]
        image_type = st.session_state.uploaded_image["type"]
        st.session_state.uploaded_image = None # Xóa ảnh khỏi trạng thái chờ

    # Nếu có ảnh mới được tải lên ngay cùng câu hỏi (ưu tiên ảnh mới)
    elif uploaded_file:
        image_bytes = uploaded_file.read()
        image_type = uploaded_file.type

    current_user_message = user_question

    # Bước 1: Tìm kiếm Context
    kb_context = search_knowledge_semantic(current_user_message)
    
    contents = []

    # Thêm ảnh vào contents (Multimodal)
    if image_bytes:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=image_type)
        contents.append(image_part)

    # Bước 2: Xây dựng Prompt (Phân biệt Có KB và Không có KB)
    if kb_context:
        # Trường hợp 1: CÓ Context (Áp dụng thẻ KB_START/KB_END nghiêm ngặt)
        full_prompt = f"""
<KB_START>
📚 KIẾN THỨC CẦN THAM KHẢO:
{kb_context}
<KB_END>

--- HỎI ĐÁP ---
Câu hỏi của học sinh:
{current_user_message}
"""
    else:
        # Trường hợp 2: KHÔNG CÓ Context
        full_prompt = f"""
Không có tài liệu tham khảo liên quan được tìm thấy.
Hãy trả lời dựa trên kiến thức nền tảng của bạn (theo Chương trình GDPT 2018).

Câu hỏi:
{current_user_message}
"""

    contents.append(full_prompt)

    # Bước 3: Gửi và Hiển thị
    with st.chat_message("Học sinh"):
        if image_bytes:
             st.markdown(f"**Bài tập Đính kèm Ảnh:**")
        st.markdown(current_user_message)
    st.session_state.messages.append({"role": "Học sinh", "content": current_user_message})

    # 📌 ĐÃ SỬA: BỔ SUNG CƠ CHẾ THỬ LẠI API (RETRY LOGIC)
    with st.spinner("⏳ Gia sư đang trả lời..."):
        reply = None
        MAX_RETRIES = 3
        RETRY_DELAY = 5

        for attempt in range(MAX_RETRIES):
            try:
                response = st.session_state.chat_session.send_message(contents)
                reply = response.text
                break
            except Exception as e:
                error_message = str(e)
                if "503 UNAVAILABLE" in error_message or "overloaded" in error_message:
                    if attempt < MAX_RETRIES - 1:
                        st.warning(f"⚠️ API bị quá tải (Lần {attempt + 1}/{MAX_RETRIES}). Đang thử lại sau {RETRY_DELAY} giây...")
                        time.sleep(RETRY_DELAY)
                    else:
                        reply = f"❌ Sau {MAX_RETRIES} lần thử, API vẫn quá tải. Vui lòng thử lại sau vài phút."
                else:
                    reply = f"⚠️ Lỗi xử lý API Gemini không xác định: {type(e).__name__}: {e}. Vui lòng kiểm tra lại code hoặc API key."
                    break

        if reply is None:
             reply = "⚠️ Lỗi: Không thể nhận phản hồi từ Gemini sau nhiều lần thử."
             
    with st.chat_message("Gia Sư"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "Gia Sư", "content": reply})
    
    # Reset file uploader nếu có ảnh mới được upload (chống lặp)
    if uploaded_file is not None:
        st.session_state['file_key'] += 1
        
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
