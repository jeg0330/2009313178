import numpy as np


def extract_best_segment(similarities, segments):
    """
    유사도 배열과 각 문장의 정보를 담은 리스트를 받아서 가장 높은 유사도를 가진 구간과 해당 유사도 값을 반환합니다.

    매개변수:
      similarities (list or np.array): 각 문장과 키워드 간의 코사인 유사도 값 리스트
      segments (list of dict): 각 문장(구간)의 정보가 담긴 리스트. 각 dict는 'processed_text', 'start', 'duration' 등을 포함합니다.

    반환:
      dict: 가장 높은 유사도를 가진 구간의 정보를 담은 딕셔너리, 여기에는 'similarity' 항목도 포함됩니다.
    """
    best_index = int(np.argmax(similarities))
    segment = segments[best_index]
    segment["similarity"] = similarities[best_index]
    return segment
