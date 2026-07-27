import os
import re
import sqlite3
import json
import math
import time
import streamlit as st
from pypdf import PdfReader
import docx
import pandas as pd
from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Çok Dilli Akıllı RAG Asistanı", page_icon="🤖", layout="wide")

DB_PATH = "db/rag_database.db"
DATA_FOLDER = "data"
SIMILARITY_THRESHOLD = 0.35  # Vektör ADAY listesine girme eşiği (RRF öncesi)
RRF_K = 60                   # Reciprocal Rank Fusion sabiti (standart/robust varsayılan)
CANDIDATE_K = 20             # Vektör ve keyword aramasından alınacak aday sayısı
MAX_STREAM_SECONDS = 300      # Tekrar döngüsü/donma durumunda üretimi kesmek için üst zaman sınırı
MAX_RESPONSE_CHARS = 6000    # Sağlıklı bir yanıtın çok üstünde; tekrar döngüsü belirtisi
MAX_EMBEDDING_CACHE_SIZE = 200  # Sorgu embedding önbelleğinin üst sınırı (LRU-benzeri tahliye)

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs("db", exist_ok=True)

# --- VERİTABANI VE METADATA YARDIMCI FONKSİYONLARI ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            file_type TEXT,
            chunk_index INTEGER,
            text_chunk TEXT,
            embedding TEXT
        )
    """)
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            text_chunk, tokenize="unicode61 remove_diacritics 2"
        )
    """)
    conn.commit()
    conn.close()

def clear_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents")
    cursor.execute("DELETE FROM documents_fts")
    conn.commit()
    conn.close()

def get_indexed_files():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT file_name FROM documents")
    files = [row[0] for row in cursor.fetchall()]
    conn.close()
    return files

def get_indexed_types():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT file_type FROM documents")
    types = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()
    return types

init_db()

# --- MATEMATİKSEL BENZERLİK VE MODEL YÖNETİMİ ---
def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a * a for a in v1))
    magnitude_v2 = math.sqrt(sum(b * b for b in v2))
    if not magnitude_v1 or not magnitude_v2:
        return 0.0
    return dot_product / (magnitude_v1 * magnitude_v2)

# --- [DÜZELTME #8] get_foundry_manager artık SADECE zararsız "already
# initialized" durumunu yutuyor. Gerçek bir başlatma hatası (servis kapalı,
# bozuk kurulum vb.) artık çağırana yükseltiliyor ki try/except'i olan
# fonksiyonlar (load_llm_model, get_query_embedding, process_and_index_files)
# özenle yazılmış "Model indirme/donanım hatası alındı" mesajını gösterebilsin.
def get_foundry_manager():
    """Foundry Manager'ı güvenli şekilde başlatır veya var olanı getirir."""
    if FoundryLocalManager.instance is not None:
        return FoundryLocalManager.instance
    try:
        config = Configuration(app_name="foundry_local_rag")
        FoundryLocalManager.initialize(config)
    except FoundryLocalException:
        if FoundryLocalManager.instance is None:
            # Singleton hâlâ kurulamadıysa bu GERÇEK bir başlatma hatasıdır,
            # sessizce yutma — çağıran taraf yakalayıp kullanıcıya göstersin.
            raise
        # instance artık doluysa (ör. eşzamanlı bir rerun/thread aynı anda
        # başlatmayı tamamladıysa) bu zaten beklenen/zararsız bir durumdur.
    return FoundryLocalManager.instance

def _select_cpu_variant(model):
    """GPU/NPU varyantı indirilemezse geri dönülecek CPU varyantını bulur (varsa)."""
    for variant in model.variants:
        runtime = getattr(variant.info, "runtime", None)
        if runtime is not None and runtime.device_type == "CPU":
            return variant
    for variant in model.variants:  # runtime bilgisi yoksa isimden tahmin (yedek)
        if "cpu" in variant.id.lower():
            return variant
    return None

def _download_and_load_with_fallback(model):
    """
    Modeli indirir ve belleğe yükler. Varsayılan varyant (genelde GPU/NPU —
    Windows'ta sorunsuz ama bazı Mac donanımlarında indirilemiyor) hata
    verirse otomatik olarak CPU varyantına geçip bir kez daha dener.
    İkisi de başarısız olursa FoundryLocalException çağırana yükseltilir.
    """
    try:
        model.download()
        model.load()
    except FoundryLocalException:
        cpu_variant = _select_cpu_variant(model)
        if cpu_variant is None or cpu_variant.id == model.info.id:
            raise  # Denenecek farklı bir varyant yok, orijinal hatayı yükselt
        model.select_variant(cpu_variant)
        model.download()
        model.load()

@st.cache_resource
def load_llm_model(model_name):
    # --- [DÜZELTME #8] get_foundry_manager() ve get_model() artık try
    # bloğunun İÇİNDE — önceden bu ikisi try'ın dışındaydı, yani bir hata
    # hiç yakalanmadan ham bir Streamlit çökme ekranına yol açıyordu.
    try:
        manager = get_foundry_manager()
        llm_model = manager.catalog.get_model(model_name)
    except FoundryLocalException as e:
        print(f"⚠️ [gui.py] Foundry Local başlatma hatası (load_llm_model): {e}")
        st.error("Model indirme/donanım hatası alındı, lütfen tekrar deneyin.")
        st.stop()
    except Exception as e:
        print(f"⚠️ [gui.py] Beklenmeyen hata (load_llm_model / manager): {e}")
        st.error(f"⚠️ Model yüklenirken beklenmeyen bir hata oluştu: {e}")
        st.stop()

    if llm_model is None:
        st.error(f"❌ '{model_name}' modeli Foundry Local kataloğunda bulunamadı. Lütfen listeden başka bir model seçin.")
        st.stop()

    try:
        with st.spinner(f"⏳ '{model_name}' modeli hazırlanıyor..."):
            original_variant_id = llm_model.info.id
            _download_and_load_with_fallback(llm_model)
            if llm_model.info.id != original_variant_id:
                st.info("ℹ️ Bu cihazda GPU/NPU hızlandırma kullanılamadı, model CPU modunda çalıştırılıyor (yanıtlar biraz daha yavaş olabilir).")
    except FoundryLocalException as e:
        print(f"⚠️ [gui.py] Model indirme/yükleme hatası (load_llm_model): {e}")
        st.error("Model indirme/donanım hatası alındı, lütfen tekrar deneyin.")
        st.stop()
    except Exception as e:
        print(f"⚠️ [gui.py] Beklenmeyen hata (load_llm_model / download): {e}")
        st.error(f"⚠️ Model yüklenirken beklenmeyen bir hata oluştu: {e}")
        st.stop()

    return manager, llm_model

# --- [DÜZELTME #4] Sidebar'dan farklı bir model seçildiğinde önceki modeli
# bellekten/VRAM'den boşaltır. @st.cache_resource tek başına bunu yapmıyordu;
# her model_name için ayrı bir kayıt açıp eskisini süresiz tutuyordu.
def switch_llm_model(model_name):
    manager = get_foundry_manager()
    previous_name = st.session_state.get("current_llm_model_name")
    if previous_name is not None and previous_name != model_name:
        try:
            previous_model = manager.catalog.get_model(previous_name)
            if previous_model is not None and previous_model.is_loaded:
                previous_model.unload()
        except Exception as e:
            print(f"⚠️ [gui.py] Önceki model ({previous_name}) boşaltılırken hata (yok sayıldı): {e}")

    manager, llm_model = load_llm_model(model_name)
    st.session_state.current_llm_model_name = model_name
    return manager, llm_model

# ==============================================================================
# EMBEDDING CACHING (Sorgu Vektörü Önbellekleme Mimarisi)
# ==============================================================================
def get_query_embedding(query_text):
    """
    Sorgu vektörünü st.session_state üzerinde önbelleğe alır.
    Gerçek Cache Hit / Miss durumunu %100 doğru raporlar.
    """
    if "embedding_cache" not in st.session_state:
        st.session_state.embedding_cache = {}

    # --- CACHE HIT (Önbellekte Var) ---
    if query_text in st.session_state.embedding_cache:
        return st.session_state.embedding_cache[query_text], True

    # --- CACHE MISS (İlk Kez Hesaplamalı) ---
    # [DÜZELTME #8] get_foundry_manager()/get_model() artık try içinde.
    try:
        manager = get_foundry_manager()
        embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
        _download_and_load_with_fallback(embedding_model)
    except FoundryLocalException as e:
        print(f"⚠️ [gui.py] Foundry Local/model hatası (get_query_embedding): {e}")
        st.error("Model indirme/donanım hatası alındı, lütfen tekrar deneyin.")
        st.stop()
    except Exception as e:
        print(f"⚠️ [gui.py] Beklenmeyen hata (get_query_embedding): {e}")
        st.error(f"⚠️ Model yüklenirken beklenmeyen bir hata oluştu: {e}")
        st.stop()

    embedding_client = embedding_model.get_embedding_client()
    response = embedding_client.generate_embeddings([query_text])
    query_vector = response.data[0].embedding

    embedding_model.unload()

    # Hafızaya kaydet
    st.session_state.embedding_cache[query_text] = query_vector

    # --- [DÜZELTME #9] Sınırsız büyümeyi önlemek için basit LRU-benzeri
    # tahliye: sözlük sınırı aşarsa en eski (ilk eklenen) girdiyi sil.
    if len(st.session_state.embedding_cache) > MAX_EMBEDDING_CACHE_SIZE:
        oldest_key = next(iter(st.session_state.embedding_cache))
        del st.session_state.embedding_cache[oldest_key]

    return query_vector, False

# --- DOSYA İŞLEME VE İNDEKSLEME ---

def load_file_content(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif ext == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text])
        elif ext in [".xlsx", ".xls", ".csv"]:
            df = pd.read_csv(file_path) if ext == ".csv" else pd.read_excel(file_path)
            text = df.to_markdown(index=False)
    except Exception as e:
        st.error(f"⚠️ {file_path} okunurken hata: {e}")
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    units = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= chunk_size:
            units.append(words)
        else:
            for sent in re.split(r"(?<=[.!?])\s+", para):
                sent_words = sent.split()
                if sent_words:
                    units.append(sent_words)

    chunks, current = [], []
    for unit in units:
        if current and len(current) + len(unit) > chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.extend(unit)
        while len(current) > chunk_size:
            chunks.append(" ".join(current[:chunk_size]))
            current = current[chunk_size - overlap:] if overlap else current[chunk_size:]
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]

def process_and_index_files(uploaded_files, manager):
    # [DÜZELTME #8] get_model() artık try içinde.
    try:
        embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
        _download_and_load_with_fallback(embedding_model)
    except FoundryLocalException as e:
        print(f"⚠️ [gui.py] Foundry Local/model hatası (process_and_index_files): {e}")
        st.error("Model indirme/donanım hatası alındı, lütfen tekrar deneyin.")
        st.stop()
    except Exception as e:
        print(f"⚠️ [gui.py] Beklenmeyen hata (process_and_index_files): {e}")
        st.error(f"⚠️ Model yüklenirken beklenmeyen bir hata oluştu: {e}")
        st.stop()
    embedding_client = embedding_model.get_embedding_client()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()

    indexed_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()

    for f_idx, uploaded_file in enumerate(uploaded_files):
        file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # --- [DÜZELTME #5] Aynı isimle daha önce indekslenmiş bir dosya
        # varsa, yeni parçaları eklemeden ÖNCE eski parçalarını (hem
        # documents hem documents_fts) sil. Aksi halde güncellenen bir
        # dosya yeniden yüklendiğinde eski+yeni chunk'lar birlikte birikip
        # hybrid aramaya çelişkili/bayat içerik sokuyordu.
        cursor.execute(
            "DELETE FROM documents_fts WHERE rowid IN (SELECT id FROM documents WHERE file_name = ?)",
            (uploaded_file.name,)
        )
        cursor.execute("DELETE FROM documents WHERE file_name = ?", (uploaded_file.name,))

        text_content = load_file_content(file_path)
        if not text_content.strip():
            continue

        chunks = chunk_text(text_content, chunk_size=300, overlap=50)
        file_type = os.path.splitext(uploaded_file.name)[1].lower()
        total_chunks = len(chunks)

        # 🛡️ CPU Güvenli Paket Boyutu: batch_size = 2
        batch_size = 2

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            current_end = min(i + batch_size, total_chunks)

            status_text.info(f"⏳ `{uploaded_file.name}` işleniyor: **{current_end} / {total_chunks}** parça vektörleştirildi...")

            try:
                response = embedding_client.generate_embeddings(batch_chunks)

                for idx_in_batch, emb_obj in enumerate(response.data):
                    chunk = batch_chunks[idx_in_batch]
                    global_idx = i + idx_in_batch
                    vector = emb_obj.embedding

                    cursor.execute(
                        "INSERT INTO documents (file_name, file_type, chunk_index, text_chunk, embedding) VALUES (?, ?, ?, ?, ?)",
                        (uploaded_file.name, file_type, global_idx, chunk, json.dumps(vector))
                    )
                    doc_id = cursor.lastrowid
                    cursor.execute(
                        "INSERT INTO documents_fts(rowid, text_chunk) VALUES (?, ?)",
                        (doc_id, chunk)
                    )
                    indexed_count += 1
            except Exception as e:
                st.warning(f"⚠️ {i}-{current_end} arasındaki parçalarda hata oluştu: {e}")

        progress_bar.progress((f_idx + 1) / len(uploaded_files))

    conn.commit()
    conn.close()
    embedding_model.unload()
    status_text.empty()
    progress_bar.empty()
    return indexed_count

# --- SORGULAMA & HYBRID ARAMA MANTIĞI (BM25 + Vektör + RRF) ---
def _sanitize_fts_query(query_text):
    """Kullanıcı sorgusunu FTS5 MATCH için güvenli hale getirir (özel karakterleri kaçırır)."""
    tokens = re.findall(r"\w+", query_text, flags=re.UNICODE)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)

def _passes_context_gate(cosine_score, matched_set):
    """
    Parça bazlı esnek gatekeeper. En yüksek skora bakıp top_k'nın TAMAMINI
    ya alan ya reddeden global bir kapı yerine, her parça kendi skoruna ve
    nasıl yakalandığına göre tek tek değerlendirilir.
    """
    if cosine_score >= 0.38:
        return True
    
    #if matched_set == {"vector", "keyword"}:
    #   return True
    
    if "keyword" in matched_set and cosine_score >= 0.25:
        return True
    return False

def retrieve_context(query_text, manager, top_k=3, selected_files=None, selected_types=None):

    # 1. Aşama: SQL Dinamik Ön-Filtreleme (Pre-filtering)
    conditions = []
    params = []

    if selected_files:
        placeholders = ",".join(["?"] * len(selected_files))
        conditions.append(f"d.file_name IN ({placeholders})")
        params.extend(selected_files)

    if selected_types:
        placeholders = ",".join(["?"] * len(selected_types))
        conditions.append(f"d.file_type IN ({placeholders})")
        params.extend(selected_types)

    where_clause = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    cursor.execute(f"SELECT d.id, d.file_name, d.file_type, d.text_chunk, d.embedding FROM documents d{where_clause}", params)
    rows = cursor.fetchall()

    if not rows:
        conn.close()
        return "", [], [], 0, False

    # 2. Aşama: CACHED Sorgu Vektörü Getir
    query_vector, is_cached = get_query_embedding(query_text)

    # 3. Aşama: Vektör Adayları (kosinüs benzerliği, CANDIDATE_K aday listesine giriş eşiği)
    row_lookup = {r[0]: r for r in rows}
    vector_scored = []
    for doc_id, file_name, file_type, text_chunk, embedding_json in rows:
        doc_vector = json.loads(embedding_json)
        score = cosine_similarity(query_vector, doc_vector)
        if score >= SIMILARITY_THRESHOLD:
            vector_scored.append((doc_id, score))
    vector_scored.sort(key=lambda x: x[1], reverse=True)
    vector_candidates = vector_scored[:CANDIDATE_K]
    cosine_lookup = dict(vector_scored)

    # 4. Aşama: Keyword Adayları (FTS5 MATCH + BM25), aynı metadata filtresiyle
    keyword_candidates = []
    fts_query = _sanitize_fts_query(query_text)
    if fts_query:
        fts_sql = """
            SELECT d.id FROM documents_fts
            JOIN documents d ON d.id = documents_fts.rowid
            WHERE documents_fts MATCH ?
        """
        fts_params = [fts_query]
        if conditions:
            fts_sql += " AND " + " AND ".join(conditions)
            fts_params.extend(params)
        fts_sql += " ORDER BY bm25(documents_fts) ASC LIMIT ?"  # bm25() negatif döner, düşük=daha alakalı
        fts_params.append(CANDIDATE_K)
        try:
            cursor.execute(fts_sql, fts_params)
            keyword_candidates = [r[0] for r in cursor.fetchall() if r[0] in row_lookup]
        except sqlite3.OperationalError:
            keyword_candidates = []

    conn.close()

    # 5. Aşama: RRF (Reciprocal Rank Fusion) Füzyonu
    rrf_scores = {}
    matched_by = {}
    for rank, (doc_id, _) in enumerate(vector_candidates, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        matched_by.setdefault(doc_id, set()).add("vector")
    for rank, doc_id in enumerate(keyword_candidates, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        matched_by.setdefault(doc_id, set()).add("keyword")

    ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    # 6. Aşama: GERÇEK KOSİNÜS HESABI
    for doc_id in ranked_ids:
        if doc_id not in cosine_lookup:
            embedding_json = row_lookup[doc_id][4]
            doc_vector = json.loads(embedding_json)
            cosine_lookup[doc_id] = cosine_similarity(query_vector, doc_vector)

    # 7. Aşama: PARÇA BAZLI ESNEK GATEKEEPER + BAĞLAM İNŞASI
    top_results = []
    context_parts = []
    unique_sources = set()
    for doc_id in ranked_ids:
        _, file_name, file_type, text_chunk, _ = row_lookup[doc_id]
        cosine_score = cosine_lookup[doc_id]
        matched_set = matched_by[doc_id]
        included = _passes_context_gate(cosine_score, matched_set)

        top_results.append((
            file_name, file_type, text_chunk,
            cosine_score, rrf_scores[doc_id], matched_set, included
        ))

        if included:
            context_parts.append(f"[{file_name}] {text_chunk}")
            unique_sources.add(file_name)

    context_text = "\n\n".join(context_parts)
    unique_sources = list(unique_sources)

    return context_text, unique_sources, top_results, len(rows), is_cached

# --- SOL MENÜ (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Sistem Ayarları")

    MODEL_OPTIONS = {
        "⚡ Qwen 2.5 (1.5B) - Hızlı & Hafif": "qwen2.5-1.5b",
        "🚀 Qwen 3.5 (2B) - Yeni Nesil & Dengeli": "qwen3.5-2b",
        "🧠 Phi-4 Mini - Microsoft Akıl Yürütme": "phi-4-mini",
        "🔬 DeepSeek R1 (7B) - Derin Analiz & Mantık": "deepseek-r1-7b",
        "💻 Qwen Coder (1.5B) - Kod & Teknik Uzman": "qwen2.5-coder-1.5b"
    }

    selected_label = st.selectbox(
        "🤖 LLM Modeli Seçin:",
        options=list(MODEL_OPTIONS.keys()),
        index=0
    )

    selected_model_name = MODEL_OPTIONS[selected_label]
    # [DÜZELTME #4] load_llm_model yerine switch_llm_model — model değişimi
    # sırasında önceki modeli otomatik boşaltır.
    manager, llm_model = switch_llm_model(selected_model_name)

    st.divider()

    st.header("📁 Doküman Yükleme")
    uploaded_files = st.file_uploader(
        "PDF, Word, Excel veya TXT yükleyin:",
        type=["pdf", "txt", "docx", "xlsx", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Dosyaları İndeksle"):
        with st.spinner("Dosyalar işleniyor..."):
            count = process_and_index_files(uploaded_files, manager)
            st.success(f"✅ {len(uploaded_files)} dosya ({count} parça %100 eksiksiz eklendi!)")
            st.rerun()

    st.divider()

    st.header("🔍 Metadata Filtreleme (Pre-filter)")
    indexed_files = get_indexed_files()
    indexed_types = get_indexed_types()

    selected_files = st.multiselect(
        "📄 Özel Dosya Seçimi:",
        options=indexed_files
    )

    selected_types = st.multiselect(
        "🏷️ Dosya Türü Filtresi:",
        options=indexed_types
    )

    if indexed_files:
        if st.button("🗑️ Veritabanını Sıfırla"):
            clear_db()
            # --- [DÜZELTME #9] Önceden burada st.cache_data.clear() vardı;
            # proje artık @st.cache_data hiç kullanmadığı için bu çağrı
            # HİÇBİR ŞEY yapmıyordu ve gerçek embedding_cache asla
            # temizlenmiyordu. Şimdi gerçek önbellek doğrudan sıfırlanıyor.
            st.session_state.embedding_cache = {}
            st.warning("Veritabanı ve önbellek temizlendi.")
            st.rerun()

# --- ANA SOHBET EKRANI ---
st.title("🤖 Kurumsal Çok Dilli RAG Asistanı")
st.caption(f"Aktif Model: `{selected_model_name}` | ⚡ Embedding Caching Aktif | 🔀 Hybrid Search (BM25 + Vektör) Aktif")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.caption(f"📚 **Kullanılan Kaynaklar:** `{', '.join(message['sources'])}` | ⚡ **Sorgu Vektörü:** Önbellekten (RAM) getirildi.")

if user_query := st.chat_input("Filtrelenmiş belgeler hakkında soru sorun..."):

    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = []

        context, sources, details, pre_filtered_count, is_cached = retrieve_context(
            user_query,
            manager,
            top_k=3,
            selected_files=selected_files,
            selected_types=selected_types
        )

        if pre_filtered_count == 0:
            full_response = "⚠️ **Metadata Filtre Uyarısı:** Seçtiğiniz filtreye uygun doküman bulunamadı."
            message_placeholder.markdown(full_response)
        elif not context:
            full_response = "Bu bilgi belgelerimde bulunmuyor."
            message_placeholder.markdown(full_response)
        else:
            client = llm_model.get_chat_client()
            system_instruction = (
                "Sen strictly bağlama bağlı bir asistanısın. Sana verilen Bağlam metni "
                "kullanıcının Sorusunu DOĞRUDAN cevaplamıyorsa, kesinlikle bağlamdaki "
                "metinleri özetleme veya sunma. Sadece tek bir cümle yaz: "
                "'Bu bilgi belgelerimde bulunmuyor.'"
            )

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {user_query}"}
            ]

            # --- [DÜZELTME #6] HATA YAKALAMALI GÜVENLİ AKIŞ ---
            try:
                stream_started_at = time.monotonic()
                for chunk in client.complete_streaming_chat(messages):
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                        if time.monotonic() - stream_started_at > MAX_STREAM_SECONDS:
                            full_response += "\n\n⚠️ *(Yanıt üretimi zaman aşımına uğradığı için durduruldu.)*"
                            break
                        if len(full_response) > MAX_RESPONSE_CHARS:
                            full_response += "\n\n⚠️ *(Yanıt beklenenden çok uzadığı — olası tekrar döngüsü — için durduruldu.)*"
                            break
            except Exception as e:
                print(f"⚠️ [gui.py] LLM streaming hatası: {e}")
                full_response += "\n\n⚠️ *(Yanıt üretimi zaman aşımı veya sistem iptali nedeniyle yarıda kesildi.)*"

            message_placeholder.markdown(full_response)

            cache_status_msg = "⚡ **Sorgu Vektörü:** Önbellekten (RAM) getirildi." if is_cached else "⚙️ **Sorgu Vektörü:** Model tarafından yeni üretildi."
            if sources:
                st.caption(f"📚 **Kullanılan Kaynaklar:** `{', '.join(sources)}` | {cache_status_msg}")

        if pre_filtered_count > 0 and details:
            with st.expander("🔍 Metadata & Önbellek Detayları"):
                if is_cached:
                    st.success("⚡ **Embedding Cache Durumu:** Sorgu vektörü RAM önbelleğinden anında çekildi (0ms).")
                else:
                    st.info("⚙️ **Embedding Cache Durumu:** Sorgu ilk kez sorulduğu için vektör model tarafından hesaplandı ve önbelleğe alındı.")
                    st.write(f"🎯 **Ön Filtreleme:** **{pre_filtered_count} parça** hybrid aramaya (BM25 + Vektör) tabi tutuldu.")
                st.divider()
                badge_map = {
                    frozenset({"vector"}): "🧭 Anlamsal",
                    frozenset({"keyword"}): "🔎 Anahtar Kelime",
                    frozenset({"vector", "keyword"}): "🔀 Her ikisi",
                }
                for file_name, file_type, chunk, cosine_score, rrf_score, matched, included in details:
                    badge = badge_map.get(frozenset(matched), "❓")
                    gate_note = "" if included else " · ⛔ eşik altı, bağlama alınmadı"
                    st.write(f"📄 **{file_name}** | {badge} | 🎯 Kosinüs: `%{cosine_score*100:.1f}` | RRF: `{rrf_score:.4f}`{gate_note}")
                    st.info(chunk)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources
    })
