import os
import json
import re
import sqlite3
import math
from pypdf import PdfReader
import docx
import pandas as pd
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_PATH = "db/rag_database.db"
DATA_FOLDER = "data"

# 1. Farklı Dosya Formatlarını Okuyan Yardımcı Fonksiyon
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
            if ext == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            # Excel tablosunu Markdown formatına dönüştürüyoruz
            text = df.to_markdown(index=False)
    except Exception as e:
        print(f"⚠️ {file_path} okunurken hata oluştu: {e}")
        
    return text

# 2. Metin Parçalayıcı (Akıllı Chunking: önce paragraf, sığmazsa cümle sınırı)
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

# 3. Gelişmiş Veritabanı Kurulumu (Metadata + FTS5 Destekli)
def setup_database():
    os.makedirs("db", exist_ok=True)
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
    cursor.execute("DELETE FROM documents")       # Eski verileri temizle
    cursor.execute("DELETE FROM documents_fts")   # FTS5 indeksini de senkron temizle
    conn.commit()
    conn.close()

# 4. Ana Veri İşleme Akışı
def process_documents():
    setup_database()
    
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"📁 '{DATA_FOLDER}' klasörü oluşturuldu. Lütfen dosyalarınızı buraya atın.")
        return

    files = [f for f in os.listdir(DATA_FOLDER) if not f.startswith(".")]
    total_chunks = 0

    for file_name in files:
        file_path = os.path.join(DATA_FOLDER, file_name)
        file_type = os.path.splitext(file_name)[1].lower()
        
        print(f"📄 İşleniyor ({file_type}): {file_name}...")
        text_content = load_file_content(file_path)
        
        if not text_content.strip():
            print(f"   ⚠️ İçerik bulunamadı veya okunamadı: {file_name}")
            continue
            
        chunks = chunk_text(text_content)
        
        for idx, chunk in enumerate(chunks):
            response = embedding_client.generate_embeddings([chunk])
            vector = response.data[0].embedding
            vector_json = json.dumps(vector)
            
            cursor.execute(
                "INSERT INTO documents (file_name, file_type, chunk_index, text_chunk, embedding) VALUES (?, ?, ?, ?, ?)",
                (file_name, file_type, idx, chunk, vector_json)
            )
            doc_id = cursor.lastrowid
            cursor.execute(
                "INSERT INTO documents_fts(rowid, text_chunk) VALUES (?, ?)",
                (doc_id, chunk)
            )
            total_chunks += 1

    conn.commit()
    conn.close()
    embedding_model.unload()
    print(f"\n✅ BAŞARILI: Toplam {len(files)} dosya, {total_chunks} parça halinde veritabanına indekslendi (vektör + FTS5)!")

if __name__ == "__main__":
    process_documents()
