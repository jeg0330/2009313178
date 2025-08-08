from sentence_transformers import SentenceTransformer

# 최신 사전학습 모델을 사용하여 모델을 로드합니다.
model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_embeddings(text_list):
    """
    주어진 텍스트 리스트를 임베딩 벡터 리스트로 변환합니다.

    매개변수:
      text_list (list of str): 임베딩을 생성할 텍스트 리스트

    반환:
      list: 각 텍스트에 대한 임베딩 벡터 리스트
    """
    embeddings = model.encode(text_list)
    return embeddings


def generate_embedding(text):
    """
    단일 텍스트에 대해 임베딩 벡터를 생성합니다.

    매개변수:
      text (str): 임베딩을 생성할 텍스트

    반환:
      vector: 해당 텍스트에 대한 임베딩 벡터
    """
    return model.encode([text])[0]
