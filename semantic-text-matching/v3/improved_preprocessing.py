import re
import unicodedata


def normalize_text(text):
    """
    텍스트를 정규화합니다. 유니코드 정규화를 수행하고 불필요한 공백을 제거합니다.
    """
    # 유니코드 정규화 (NFKC: 호환성 정규화)
    text = unicodedata.normalize('NFKC', text)
    # 연속된 공백을 하나로 변환
    text = re.sub(r'\s+', ' ', text)
    # 좌우 공백 제거
    return text.strip()


def preprocess_text(text):
    """
    입력 텍스트를 정규화하고, 불필요한 특수문자를 제거하되 의미 파악에 필요한 일부 문장부호는 유지합니다.
    """
    # 1. 텍스트 정규화
    text = normalize_text(text)

    # 2. 소문자화 (영문만 해당)
    text = text.lower()

    # 3. 불필요한 특수 문자 제거 (한글, 영문, 숫자, 기본 문장부호만 유지)
    # 마침표, 쉼표, 물음표, 느낌표, 따옴표, 콜론, 세미콜론, 괄호 등은 유지
    cleaned_text = re.sub(r'[^\w\s.,!?:;\'\"\(\)\-가-힣]', '', text)

    return cleaned_text


def split_into_sentences(text):
    """
    텍스트를 문장 단위로 분할합니다.
    한국어와 영어의 문장 종결 패턴을 모두 고려합니다.
    """
    # 마침표, 물음표, 느낌표 등을 기준으로 문장 분할
    # 단, 숫자 뒤의 마침표나 약어의 마침표는 문장 분할에서 제외

    # 다음의 패턴으로 문장 분할:
    # 1. 문장 종결 부호 (.!?) 뒤에 공백이나 문장의 끝인 경우
    # 2. 한국어 문장 종결 어미 (예: '다.', '까?', '요.' 등)
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z가-힣])|(?<=다\.)|(?<=까\?)|(?<=요\.)'

    sentences = re.split(sentence_pattern, text)

    # 좌우 공백 제거 및 빈 문자열 제거
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return sentences


def merge_subtitle_segments(subtitles, max_gap=1.0):
    """
    시간 간격이 가까운 자막을 하나로 병합하여 문맥을 보존합니다.
    max_gap: 자막 간 최대 허용 시간 간격(초)
    """
    if not subtitles:
        return []

    merged = []
    current_segment = subtitles[0].copy()

    for i in range(1, len(subtitles)):
        current_end = current_segment["start"] + current_segment["duration"]
        next_start = subtitles[i]["start"]

        # 시간 간격이 max_gap보다 작으면 병합
        if next_start - current_end <= max_gap:
            # 텍스트 병합
            current_segment["text"] += " " + subtitles[i]["text"]
            # 지속 시간 업데이트
            current_segment["duration"] = (subtitles[i]["start"] + subtitles[i]["duration"]) - current_segment["start"]
        else:
            # 현재 세그먼트 저장하고 새로운 세그먼트 시작
            merged.append(current_segment)
            current_segment = subtitles[i].copy()

    # 마지막 세그먼트 추가
    merged.append(current_segment)

    return merged


def process_subtitles_json(json_data, merge_segments=True):
    """
    자막 JSON 데이터를 처리하여 개선된 전처리 결과를 반환합니다.

    json_data: {
        "video_id": "video_id",
        "subtitles": [
            { "text": "subtitle text", "start": 2.12, "duration": 6.479 },
            ...
        ]
    }

    merge_segments: 인접한 자막을 병합할지 여부
    """
    subtitles = json_data.get("subtitles", [])

    # 필요한 경우 인접한 자막 세그먼트 병합
    if merge_segments:
        subtitles = merge_subtitle_segments(subtitles)

    processed_segments = []

    for subtitle in subtitles:
        original_text = subtitle.get("text", "")
        start = subtitle.get("start", 0)
        duration = subtitle.get("duration", 0)

        # 정규화 및 전처리 수행
        normalized = normalize_text(original_text)
        preprocessed = preprocess_text(original_text)

        # 문장 분할 (한 자막 항목 내에 여러 문장이 있을 수 있음)
        sentences = split_into_sentences(preprocessed)

        # 각 문장을 개별 구간으로 저장 (추가 메타데이터 포함)
        for sentence in sentences:
            processed_segments.append({
                "processed_text": sentence,
                "normalized_text": normalized,
                "original_text": original_text,
                "start": start,
                "duration": duration
            })

    return processed_segments
