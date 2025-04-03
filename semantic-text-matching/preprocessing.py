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