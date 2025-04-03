import json
from preprocessing import process_subtitles_json


def main():
    # JSON 파일에서 자막 데이터를 불러오는 예시
    # file_name = input("자막 JSON 파일 경로를 입력하세요 (예: subtitles.json): ")
    file_name = "gKEzL3pn1VA.json"  # 예시 파일 이름
    with open(file_name, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 전처리 및 문장 분할 수행 (원본 메타데이터 유지)
    processed_segments = process_subtitles_json(json_data)

    # 결과 출력 (각 구간의 전처리된 텍스트와 원본 메타데이터 확인)
    for seg in processed_segments:
        print(f"Processed: {seg['processed_text']}, Start: {seg['start']}, Duration: {seg['duration']}")


if __name__ == '__main__':
    main()