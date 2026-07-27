import sqlite3
import json
import math
import re
from foundry_local_sdk import Configuration, FoundryLocalManager

SIMILARITY_THRESHOLD = 0.35
RRF_K = 60
CANDIDATE_K = 20

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a * a for a in v1))
    magnitude_v2 = math.sqrt(sum(b * b for b in v2))
    if not magnitude_v1 or not magnitude_v2:
        return 0.0
    return dot_product / (magnitude_v1 * magnitude_v2)

def _sanitize_fts_query(query_text):
    tokens = re.findall(r"\w+", query_text, flags=re.UNICODE)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)

def hybrid_search(query_text, top_k=2):
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.load()

    embedding_client = embedding_model.get_embedding_client()
    response = embedding_client.generate_embeddings([query_text])
    query_vector = response.data[0].embedding

    embedding_model.unload()

    conn = sqlite3.connect("db/rag_database.db")
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()

    cursor.execute("SELECT id, text_chunk, embedding FROM documents")
    rows = cursor.fetchall()

    text_lookup = {}
    embedding_lookup = {}
    vector_scored = []
    for doc_id, text_chunk, embedding_json in rows:
        text_lookup[doc_id] = text_chunk
        embedding_lookup[doc_id] = embedding_json
        doc_vector = json.loads(embedding_json)
        score = cosine_similarity(query_vector, doc_vector)
        if score >= SIMILARITY_THRESHOLD:
            vector_scored.append((doc_id, score))
    vector_scored.sort(key=lambda x: x[1], reverse=True)
    vector_candidates = vector_scored[:CANDIDATE_K]
    cosine_lookup = dict(vector_scored)

    keyword_candidates = []
    fts_query = _sanitize_fts_query(query_text)
    if fts_query:
        try:
            cursor.execute(
                """
                SELECT documents_fts.rowid FROM documents_fts
                WHERE documents_fts MATCH ?
                ORDER BY bm25(documents_fts) ASC
                LIMIT ?
                """,
                (fts_query, CANDIDATE_K),
            )
            keyword_candidates = [r[0] for r in cursor.fetchall()]
            for doc_id in keyword_candidates:
                if doc_id not in text_lookup:
                    cursor.execute("SELECT text_chunk, embedding FROM documents WHERE id = ?", (doc_id,))
                    found = cursor.fetchone()
                    if found:
                        text_lookup[doc_id] = found[0]
                        embedding_lookup[doc_id] = found[1]
        except sqlite3.OperationalError:
            keyword_candidates = []

    conn.close()

    rrf_scores = {}
    matched_by = {}
    for rank, (doc_id, _) in enumerate(vector_candidates, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        matched_by.setdefault(doc_id, set()).add("vector")
    for rank, doc_id in enumerate(keyword_candidates, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        matched_by.setdefault(doc_id, set()).add("keyword")

    ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    # --- [DÜZELTME] GERÇEK KOSİNÜS HESABI (gui.py ile senkron) ---
    for doc_id in ranked_ids:
        if doc_id not in cosine_lookup and doc_id in embedding_lookup:
            doc_vector = json.loads(embedding_lookup[doc_id])
            cosine_lookup[doc_id] = cosine_similarity(query_vector, doc_vector)

    results = []
    for doc_id in ranked_ids:
        text_chunk = text_lookup.get(doc_id, "")
        results.append((text_chunk, cosine_lookup.get(doc_id, 0.0), rrf_scores[doc_id], matched_by[doc_id]))
    return results

def main():
    query = "Yapay zekanın tanımı nedir ve hangi görevleri yapar?"

    print(f"🔍 Soru Soruluyor: '{query}'")
    print("🧠 Hybrid arama yapılıyor (BM25 + Vektör + RRF)...")

    top_results = hybrid_search(query, top_k=2)

    badge_map = {
        frozenset({"vector"}): "🧭 Anlamsal",
        frozenset({"keyword"}): "🔎 Anahtar Kelime",
        frozenset({"vector", "keyword"}): "🔀 Her ikisi",
    }

    print("\n🎯 EN YAKIN BULUNAN BELGE PARÇALARI:")
    for i, (text, cosine_score, rrf_score, matched) in enumerate(top_results, 1):
        badge = badge_map.get(frozenset(matched), "❓")
        print(f"\n[{i}] {badge} | Kosinüs: {cosine_score:.4f} | RRF: {rrf_score:.4f}")
        print(f"Metin: {text}")
        print("-" * 50)

if __name__ == "__main__":
    main()
