import sqlite3

conn = sqlite3.connect("db/rag_database.db")
cursor = conn.cursor()

cursor.execute("SELECT file_name, file_type, COUNT(*) FROM documents GROUP BY file_name, file_type")
rows = cursor.fetchall()

print("\n=== 📊 VERİTABANINDAKİ DOSYALAR VE PARÇA (CHUNK) SAYILARI ===")
if not rows:
    print("❌ Veritabanı şu an tamamen boş!")
else:
    for file_name, file_type, count in rows:
        print(f"📁 Dosya: {file_name} | Tür: {file_type} | Vektör Parçası: {count} adet")

conn.close()