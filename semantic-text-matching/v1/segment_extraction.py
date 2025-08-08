import numpy as np


def extract_best_segment(similarities, segments, similarity_threshold=0.5, max_gap=1.0):
    """
    유사도 배열과 각 구간 정보를 받아, 가장 높은 유사도를 가진 구간과
    연속해서 유사도가 높은 구간들을 묶어 반환합니다.

    매개변수:
      similarities (list or np.array): 각 구간과 키워드 간 코사인 유사도 값 리스트
      segments (list of dict): 각 구간의 정보. 각 dict는 'processed_text', 'start', 'duration' 등 포함
      similarity_threshold (float): 인접 구간을 포함할 최소 유사도 기준
      max_gap (float): 인접 구간을 그룹화할 최대 시간 간격(초)

    반환:
      dict: 묶인 구간의 정보를 포함하며, 'similarity' 항목에는 그룹 내 최고 유사도 값 포함
    """
    best_index = int(np.argmax(similarities))

    # 초기 그룹: 최고 유사도 구간
    best_group = [segments[best_index]]

    # 이전 구간 확인
    i = best_index - 1
    while i >= 0:
        # 인접 시간 차이와 유사도 기준을 만족하면 그룹에 추가
        if (segments[best_index]['start'] - (segments[i]['start'] + segments[i]['duration']) <= max_gap) and (
                similarities[i] >= similarity_threshold):
            best_group.insert(0, segments[i])
            i -= 1
        else:
            break

    # 이후 구간 확인
    i = best_index + 1
    while i < len(segments):
        current_end = best_group[-1]['start'] + best_group[-1]['duration']
        if (segments[i]['start'] - current_end <= max_gap) and (similarities[i] >= similarity_threshold):
            best_group.append(segments[i])
            i += 1
        else:
            break

    # 그룹화된 구간 결합: 텍스트 결합, 시작 시간은 그룹 첫 구간, duration은 그룹 마지막 구간까지
    combined_segment = {
        "processed_text": " ".join([seg["processed_text"] for seg in best_group]),
        "original_text": " ".join([seg["original_text"] for seg in best_group]),
        "start": best_group[0]["start"],
        "duration": (best_group[-1]["start"] + best_group[-1]["duration"]) - best_group[0]["start"],
        "similarity": np.max(
            [similarities[i] for i in range(best_index - len(best_group) + 1, best_index + len(best_group))])
        # 그룹 내 최고 유사도
    }
    return combined_segment
