import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import openai
import os

# 환경 변수 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

# Qdrant 클라이언트 초기화
client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
)

# SentenceTransformer 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

# Qdrant에서 관련 데이터를 검색하는 함수
def search_qdrant(query, collection_name="sample_meeting_rc", top_k=5):
    query_vector = model.encode(query).tolist()
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    return [
        hit.payload["요약된발언내용"] for hit in search_results
    ]

# GPT-4o-mini로 답변 생성 함수
def generate_response(context, query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in Korean parliamentary data."},
                {"role": "assistant", "content": f"Here is some related data: {context}"},
                {"role": "user", "content": query}
            ],
            max_tokens=1000
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"답변 생성 실패: {str(e)}"

# Streamlit 애플리케이션
st.title("국정조사 챗봇")
st.markdown("Qdrant와 GPT-4를 활용하여 질문에 답변하는 국정조사 챗봇입니다.")

# 사용자 입력
user_query = st.text_input("질문을 입력하세요", "")

if user_query:
    with st.spinner("답변을 생성 중입니다..."):
        # Qdrant 검색
        related_data = search_qdrant(user_query)
        context = " ".join(related_data)

        # GPT-4 응답 생성
        response = generate_response(context, user_query)

        # 결과 표시
        st.subheader("챗봇 답변")
        st.write(response)

        # 관련 데이터 표시
        st.subheader("관련 데이터")
        for i, data in enumerate(related_data, 1):
            st.markdown(f"**관련 문서 {i}:** {data}")

