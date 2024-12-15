import os
import openai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from konlpy.tag import Okt
import streamlit as st

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

# 한국어 쿼리 전처리 함수
okt = Okt()
def preprocess_query(query):
    """
    사용자 입력 쿼리를 전처리하여 검색 친화적으로 변환
    """
    nouns = okt.nouns(query)  # 명사 추출
    return " ".join(nouns)

# 검색 결과 필터링
def filter_related_data(results, max_length=500):
    """
    검색 결과를 필터링하여 텍스트 길이 제한 내에서 가장 관련성 높은 데이터만 반환
    """
    return " ".join([result for result in results if len(result) <= max_length])

# Qdrant에서 검색
def search_qdrant(query, collection_name="sample_meeting_rc", top_k=5):
    """
    Qdrant에서 관련 데이터를 검색
    """
    processed_query = preprocess_query(query)
    query_vector = model.encode(processed_query).tolist()
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    return [
        hit.payload["요약된발언내용"] for hit in search_results
    ]

# GPT 모델 선택 함수
def select_model(query_length):
    """
    쿼리 길이에 따라 GPT-4o 또는 GPT-4o-mini 선택
    """
    if query_length > 500:
        return "gpt-4o"
    return "gpt-4o-mini"

# GPT 응답 생성 함수
def generate_response(context, query):
    """
    GPT 모델을 사용하여 응답 생성
    """
    model = select_model(len(query))
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an assistant specializing in Korean parliamentary data."},
                {"role": "user", "content": f"Using the following context, answer this query:\n\n{query}"},
                {"role": "assistant", "content": f"Relevant data: {context}"}
            ],
            max_tokens=1000
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"응답 생성 실패: {str(e)}"

# Streamlit 애플리케이션
st.title("국회회의록록 챗봇")
st.markdown("Qdrant와 GPT를 활용하여 국회 회의록 데이터를 기반으로 질문에 답변하는 챗봇입니다.")

# 사용자 입력
user_query = st.text_input("질문을 입력하세요", "")

if user_query:
    with st.spinner("답변을 생성 중입니다..."):
        # Qdrant에서 관련 데이터 검색
        try:
            related_data = search_qdrant(user_query)
            if not related_data:
                st.error("관련 데이터를 찾을 수 없습니다. 질문을 수정해 주세요.")
            else:
                # 데이터 필터링
                context = filter_related_data(related_data)

                # GPT 응답 생성
                response = generate_response(context, user_query)

                # 결과 출력
                st.subheader("챗봇 답변")
                st.write(response)

                # 관련 데이터 표시
                st.subheader("참조 데이터")
                for i, data in enumerate(related_data, 1):
                    st.markdown(f"**관련 문서 {i}:** {data}")
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
