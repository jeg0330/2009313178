import re


def preprocess_text(text):
    """
    입력 텍스트를 소문자화하고, 한글, 영문, 숫자, 공백을 제외한 문자를 제거합니다.
    """
    # 1. 소문자화
    text = text.lower()
    # 2. 정규표현식을 이용하여 불필요한 특수문자 제거 (한글, 영문, 숫자, 공백만 남김)
    cleaned_text = re.sub(r'[^a-z0-9\s가-힣]', '', text)
    return cleaned_text


def split_into_sentences(text):
    """
    전처리된 텍스트를 마침표, 느낌표, 물음표 기준으로 분할하여 문장 리스트를 반환합니다.
    """
    sentences = re.split(r'[.!?]+', text)
    # 좌우 공백 제거 및 빈 문자열 제거
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


def process_subtitles_json(json_data):
    """
    json_data: {
        "video_id": "gKEzL3pn1VA",
        "subtitles": [
            { "text": "앉아봐 계속 여가 가지고 바람이", "start": 2.12, "duration": 6.479 },
            { "text": "살랑살랑 신랑 같은 공장을 쓴다고", "start": 4.92, "duration": 6.36 }
        ]
    }

    각 자막 항목에 대해 전처리 및 문장 분할을 수행하고,
    원본 텍스트, start, duration 정보를 함께 보존하여 리스트로 반환합니다.
    """
    processed_segments = []
    subtitles = json_data.get("subtitles", [])
    for subtitle in subtitles:
        original_text = subtitle.get("text", "")
        start = subtitle.get("start", 0)
        duration = subtitle.get("duration", 0)

        # 전처리 수행
        preprocessed = preprocess_text(original_text)
        # 문장 분할 (한 자막 항목 내에 여러 문장이 있을 수 있음)
        sentences = split_into_sentences(preprocessed)

        # 각 문장을 개별 구간으로 저장 (원본 메타데이터 포함)
        for sentence in sentences:
            processed_segments.append({
                "processed_text": sentence,
                "original_text": original_text,
                "start": start,
                "duration": duration
            })
    return processed_segments


def group_contiguous_segments(segments, max_gap=1.0):
    """
    각 자막 구간(segments)을 시간 간격(max_gap) 내에 있는 경우 하나의 그룹으로 묶습니다.
    max_gap: 이전 구간의 끝과 다음 구간의 시작 사이 최대 허용 간격(초)
    반환:
      그룹화된 구간 리스트. 각 그룹은 'processed_text', 'original_text', 'start', 'duration' 등의 정보를 포함합니다.
    """
    if not segments:
        return []

    grouped_segments = []
    current_group = segments[0].copy()

    for seg in segments[1:]:
        current_end = current_group['start'] + current_group['duration']
        # 다음 구간의 시작과 이전 구간의 끝 사이 간격이 max_gap 이하인 경우 그룹화
        if seg['start'] - current_end <= max_gap:
            # 텍스트 결합 (공백으로 구분)
            current_group['processed_text'] += " " + seg['processed_text']
            current_group['original_text'] += " " + seg['original_text']
            # 그룹의 duration 업데이트: 마지막 구간의 끝까지로 계산
            current_group['duration'] = (seg['start'] + seg['duration']) - current_group['start']
        else:
            grouped_segments.append(current_group)
            current_group = seg.copy()
    grouped_segments.append(current_group)
    return grouped_segments
