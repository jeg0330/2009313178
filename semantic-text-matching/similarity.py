from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(keyword_embedding, embeddings):
    """
    키워드 임베딩과 문장 임베딩들 간의 코사인 유사도를 계산합니다.

    매개변수:
      keyword_embedding (vector): 전처리 및 임베딩된 키워드 벡터
      embeddings (list or array): 전처리된 각 문장에 대한 임베딩 벡터 리스트

    반환:
      list: 각 문장과 키워드 간의 코사인 유사도 값 리스트
    """
    # 키워드 벡터와 모든 문장 벡터의 코사인 유사도를 계산하여 1차원 배열로 반환
    similarities = cosine_similarity([keyword_embedding], embeddings)[0]
    return similarities
