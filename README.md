# 🤖 Yerel RAG Asistanı (Local RAG Assistant)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Foundry Local](https://img.shields.io/badge/Foundry--Local-SDK-0078D4.svg)](https://github.com/microsoft/foundry-local)
[![SQLite FTS5](https://img.shields.io/badge/SQLite-FTS5%20%2B%20Vector-003B57.svg)](https://www.sqlite.org/fts5.html)

Foundry Local SDK ve Streamlit tabanlı; tamamen yerel, yüksek performanslı, güvenli ve çok dilli bir Doküman Soru-Cevap (RAG) asistanı.

> **A privacy-first, multi-lingual local RAG solution powered by Qwen, DeepSeek, and Phi models — featuring SQLite FTS5 + Vector hybrid search (BM25 + RRF), embedding caching, and dynamic metadata filtering.**

---

## 🌟 Öne Çıkan Özellikler

* **🔒 %100 Yerel ve Güvenli (Privacy-First):** Verileriniz dış sunuculara veya buluta gitmez. Tüm metin parçalama, vektörleştirme ve LLM üretimi tamamen yerel cihazınızda gerçekleşir.
* **🔀 Hibrit Arama Teknolojisi (Hybrid Search):** 
  * **Metin Araması:** SQLite FTS5 tabanlı BM25 anahtar kelime eşleşmesi.
  * **Anlamsal Arama:** Kosinüs benzerliği (Cosine Similarity) vektör araması.
  * **Sıralama Füzyonu:** İki yöntemi en optimum şekilde birleştiren **RRF (Reciprocal Rank Fusion)** algoritması.
* **🧠 Esnek Model Desteği:** Tek tıkla arayüzden model değiştirme:
  * ⚡ `Qwen 2.5 (1.5B)` — Hızlı & Hafif (Varsayılan)
  * 🚀 `Qwen 3.5 (2B)` — Dengeli & Güncel
  * 🧠 `Phi-4 Mini` — Microsoft Mantıksal Akıl Yürütme
  * 🔬 `DeepSeek R1 (7B)` — Derin Analiz & Karmaşık Mantık
  * 💻 `Qwen Coder (1.5B)` — Kod & Teknik Metin Uzmanı
* **📄 Çoklu Doküman Formatı:** PDF, DOCX, XLSX, CSV ve TXT formatlarındaki belgeleri okuma ve indeksleme.
* **⚡ Akıllı Bellek Yönetimi (Embedding Caching):** Tekrarlanan sorgu vektörlerini RAM önbelleğinde tutarak 0 ms erişim süresi sağlar.
* **🛡️ Donanım Esnekliği (Hardware Fallback):** Cihazınızda GPU/NPU bulunamadığında veya uyumsuzluk yaşandığında otomatik olarak CPU moduna geçer, uygulamanın çökmesini engeller.
* **🔍 Dinamik Metadata Filtreleme:** Arama yapmadan önce dosya adı veya dosya türü bazlı SQL ön-filtreleme (Pre-filtering).

---

## 📋 Ön Gereksinimler (Prerequisites)

Uygulamayı çalıştırmadan önce sisteminizde aşağıdakilerin hazır olması gerekir:

1. **Python 3.10 veya üzeri**
2. **Microsoft Foundry Local Runtime / Service:** Yerel modellerin bilgisayarınızda çalışmasını sağlayan arka plan servisi.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone [https://github.com/mBeratKeremAydin/yerel_rag_asistani.git](https://github.com/mBeratKeremAydin/yerel_rag_asistani.git)
cd yerel_rag_asistani


2. Sanal Ortam Oluşturun ve Aktif Edin (Önerilir)Bash# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

3. Bağımlılıkları Yükleyin
Bashpip install -r requirements.txt

4. Uygulamayı Başlatın
Bashstreamlit run gui.py

🛠️ Foundry Local & Model HazırlığıUygulama ihtiyaç duyduğu modelleri otomatik olarak indirmeye çalışır. Ancak ilk çalıştırmadaki yavaşlıktan veya olası ağ kesintilerinden kaçınmak için gerekli modelleri terminalinizden (CMD / PowerShell / Mac Terminal) önceden indirebilirsiniz:Gerekli Modelleri Elle İndirme (CLI):Bash# 1. Embedding Modelini İndirin (ZORUNLU - Vektör Arama İçin)
foundry-local download qwen3-embedding-0.6b

# 2. Varsayılan LLM Modelini İndirin (Metin Üretimi İçin)
foundry-local download qwen2.5-1.5b
İpucu: Diğer LLM modellerini kullanmak isterseniz foundry-local download <model-adi> komutuyla (örneğin foundry-local download deepseek-r1-7b) önceden indirebilirsiniz.

## ❓ Sık Karşılaşılan Sorunlar ve Çözümleri (Troubleshooting)

* **❌ "Model indirme/donanım hatası alındı"**
  * **Olası Nedeni:** Embedding veya LLM modeli indirilirken internet koptu veya dosya bozuldu.
  * **Çözüm:** Terminalde `foundry-local download qwen3-embedding-0.6b` çalıştırarak modelin %100 indiğinden emin olun.

* **❌ "Foundry Local başlatma hatası"**
  * **Olası Nedeni:** Foundry Local servis uygulaması kapalı.
  * **Çözüm:** Windows'ta Başlat menüsünden Foundry Local servisini başlatın veya macOS'ta arka plan servisini kontrol edin.

* **❌ "Excel okunurken hata"**
  * **Olası Nedeni:** `.xlsx` dosyaları için gerekli okuyucu eksik.
  * **Çözüm:** `pip install openpyxl` komutuyla kütüphaneyi yükleyin (`requirements.txt` içinde mevcuttur).


📁 Proje YapısıPlaintextyerel_rag_asistani/
├── gui.py              # Main Streamlit Uygulaması (Arayüz + Hybrid RAG Mantığı)
├── requirements.txt    # Python kütüphane bağımlılıkları
├── .gitignore          # Git tarafından takip edilmeyecek dosya kuralları
├── README.md           # Proje tanıtım ve kullanım kılavuzu
├── data/               # Yüklenen dokümanların tutulduğu klasör (Otomatik oluşur)
└── db/                 # SQLite FTS5 veritabanı klasörü (Otomatik oluşur)
📜 LisansBu proje açık kaynaklıdır. İstediğiniz gibi geliştirebilir, kendi projelerinizde kaynak göstererek kullanabilirsiniz.