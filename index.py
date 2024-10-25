import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
import os
from dotenv import load_dotenv

# Çevresel değişkenleri yükleme (OpenAI API anahtarı)
load_dotenv()
openai.api_key = os.getenv("openai_apikey")

# Qdrant bağlantısı (yerel bir instance kullanıyorsanız localhost)
client = QdrantClient(host="127.0.0.1", port=6333)
collection_name = "yargitay_kararlar"

# Verilen metni OpenAI GPT-4 modeli kullanarak embedding vektörüne dönüştüren fonksiyon
def get_embedding_from_openai(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"  # GPT-4 kullanmak için uygun embedding modelini seçiyoruz
    )
    return response['data'][0]['embedding']

# Qdrant'ta embedding araması yapan fonksiyon
def search_in_qdrant(query_text, top_k=3):
    try:
        # OpenAI'den metin için embedding al
        query_embedding = get_embedding_from_openai(query_text)

        # Qdrant'ta embedding'lere en yakın sonuçları bulma
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k  # En yakın top_k sonuçları döndür
        )

        # Sonuçları liste olarak döndür
        results = []
        for result in search_result:
            results.append(result.payload['davakonusu'])  # 'davakonusu' sütunundaki metni alıyoruz

        return results

    except Exception as e:
        print(f"Bir hata oluştu: {e}")
        return []

# OpenAI GPT-4 chat modeli ile gelen sonuçlardan soruya cevap oluşturma
def generate_answer_from_gpt(question, documents):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides legal insights based on court decisions."},
        {"role": "user", "content": f"Soru: {question}\n\nAşağıda yer alan yargı kararlarına göre soruya bir yanıt oluştur.hukuk uzmanından yardım almak en doğru yol olacaktır gibi cümleler kurmamalısın çünkü uzman sensin.En son istediği yanıtı tam alamadığını düşünerek detayları soracağın birer cümlelik soru ifadeleri içeren 2 öneri yapmanı rica ediyorum.Bu önerileriler önüne sayısal ifadeler olmadan sadece ONERI:: ibaresi eklersen parse edebilirim.Örneğin ONERI:: Farklı yargı kararlarını araştırmamı ister misin? , gibi. "}
    ]
    
    for i, doc in enumerate(documents, start=1):
        messages.append({"role": "assistant", "content": f"Yargı Kararı {i}: {doc}"})
    
    messages.append({"role": "user", "content": "Bu bilgilere dayanarak, soruya uygun bir hukuki değerlendirme yap."})

    # GPT-4 ile chat modunda istek gönderme
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=500,
        temperature=0.4
    )

    return response['choices'][0]['message']['content'].strip()

# Arama yapmak istediğiniz metni buraya yazın
search_query = "boşanma davasından sonra erkek nafaka alabilir mi?"
yargitay_kararlari = search_in_qdrant(search_query)

if yargitay_kararlari:
    cevap = generate_answer_from_gpt(search_query, yargitay_kararlari)
    print("GPT-4 Cevap:", cevap)
else:
    print("Yargıtay kararları bulunamadı.")
