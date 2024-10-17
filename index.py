from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

# Çevresel değişkenleri yükleme (OpenAI API anahtarı)
load_dotenv()
openai.api_key = os.getenv("openai_apikey")

# Qdrant bağlantısı (yerel bir instance kullanıyorsanız localhost)
client = QdrantClient(host="4.182.180.125", port=6333)
collection_name = "yargitay_kararlar"

# Flask uygulamasını başlatma
app = Flask(__name__)

# CORS sadece belirli yollar ve kökenler için izin verir
CORS(app, resources={r"/search": {"origins": "*"}})

# Verilen metni OpenAI GPT-4 modeli kullanarak embedding vektörüne dönüştüren fonksiyon
def get_embedding_from_openai(text):
    try:
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002"  # Embedding oluşturmak için uygun model
        )
        return response['data'][0]['embedding']
    except Exception as e:
        return {"error": f"Embedding oluşturulurken bir hata oluştu: {e}"}

# OpenAI kullanarak kararların genel özetini ve avukata tavsiyeyi oluşturma fonksiyonu
# OpenAI kullanarak kararların genel özetini ve avukata tavsiyeyi oluşturma fonksiyonu
def generate_combined_summary_and_advice(documents):
    try:
        # Karar metinlerini birleştir
        combined_documents = "\n\n".join(documents)
       # Karar metinlerini kontrol etmek için

        messages = [
            {"role": "system", "content": "Sen bir avukatsın."},
            {"role": "user", "content": f"Bu yargı kararlarını özetleyin ve bir avukata bu konuda ne tavsiye edersiniz?\n\nYargı kararları: {combined_documents}"}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",  # Ya da "gpt-4-turbo" kullanabilirsiniz
            messages=messages,
            max_tokens=800
        )
        
        # OpenAI'den gelen cevabı döndür
        summary_and_advice = response['choices'][0]['message']['content'].strip()
        return summary_and_advice
    except Exception as e:
        return {"error": f"Özet ve tavsiye oluşturulurken bir hata oluştu: {e}"}

      
    except Exception as e:
        return {"error": f"Özet ve tavsiye oluşturulurken bir hata oluştu: {e}"}

# Qdrant'ta embedding araması yapan fonksiyon
def search_in_qdrant(query_text, top_k=3):
    try:
        # OpenAI'den metin için embedding al
        query_embedding = get_embedding_from_openai(query_text)

        if isinstance(query_embedding, dict) and "error" in query_embedding:
            return [], query_embedding["error"]

        # Qdrant'ta embedding'lere en yakın sonuçları bulma
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k  # En yakın top_k sonuçları döndür
        )

        # Sonuçları liste olarak döndür ve tüm dokümanları birleştir
        results = []
        documents = []
        for result in search_result:
            document_text = result.payload['davakonusu']
            documents.append(document_text)  # Tüm dokümanları topluyoruz
            
            results.append({
                "id": result.id,
                "score": result.score,
                "document": document_text
            })

        # Tüm kararlar için bir özet ve tavsiye oluştur
        combined_summary_and_advice = generate_combined_summary_and_advice(documents)
        
        return results, combined_summary_and_advice

    except Exception as e:
        return [], f"Bir hata oluştu: {e}"

# API endpoint: /search

@app.route("/search", methods=["POST"])
def search():
    try:
        # JSON istek verilerini al
        data = request.get_json()
        query_text = data.get("query_text")
        top_k = data.get("top_k", 3)  # Varsayılan top_k değeri 3

        # Qdrant'ta arama yap ve sonuçları getir
        results, combined_summary_and_advice = search_in_qdrant(query_text, top_k)

        # Eğer bir hata varsa
        if isinstance(combined_summary_and_advice, dict) and "error" in combined_summary_and_advice:
            return jsonify({"error": combined_summary_and_advice["error"]}), 500

        # Yargı kararlarının sayısını belirleyelim
        yargicount = len(results)
        
        # Sonucu JSON formatında döndürelim
        return jsonify({
            "data": {
                "count": yargicount,
                "results": results,
                "ozet": combined_summary_and_advice  # Özet ve tavsiyeyi döndürüyoruz
            }
        })
    
    except Exception as e:
        return jsonify({"error": f"Bir hata oluştu: {e}"}), 500

# Uygulamayı başlatma
if __name__ == "__main__":
    app.run(debug=True, port=5000)

