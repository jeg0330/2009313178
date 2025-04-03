### import
import json

from embedding import generate_embeddings, generate_embedding
from preprocessing import process_subtitles_json, preprocess_text
from segment_extraction import extract_best_segment
from similarity import calculate_similarity

### JSON 파일에서 자막 데이터를 불러오는 예시
# file_name = input("자막 JSON 파일 경로를 입력하세요 (예: subtitles.json): ")
file_name = "gKEzL3pn1VA.json"  # 예시 파일 이름
with open(file_name, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

### 전처리 및 문장 분할 수행 (원본 메타데이터 유지)
processed_segments = process_subtitles_json(json_data)

# 결과 출력 (각 구간의 전처리된 텍스트와 원본 메타데이터 확인)
for seg in processed_segments:
    print(f"Processed: {seg['processed_text']}, Start: {seg['start']}, Duration: {seg['duration']}")

### 사용자로부터 키워드 입력 및 전처리
# user_keyword = input("키워드를 입력하세요: ")
user_keyword = "제품 소독"
processed_keyword = preprocess_text(user_keyword)

### 임베딩 생성 단계: 각 문장과 키워드에 대해 임베딩 벡터 생성
sentences = [seg['processed_text'] for seg in processed_segments]
sentence_embeddings = generate_embeddings(sentences)
keyword_embedding = generate_embedding(processed_keyword)

print("\n키워드 임베딩 생성 완료:")
print(keyword_embedding)

### 유사도 측정 단계: 키워드와 각 문장 임베딩 간 코사인 유사도 계산
similarities = calculate_similarity(keyword_embedding, sentence_embeddings)

# 유사도 측정 결과 출력 (디버그용)
print("\n유사도 측정 결과:")
for idx, sim in enumerate(similarities):
    print(f"문장 {idx + 1}: 유사도 = {sim}")

### 최적 구간 추출 후 결과 출력 예시
best_segment = extract_best_segment(similarities, processed_segments)
print("\n키워드와 가장 유사한 구간:")
print(f"Processed: {best_segment['processed_text']}, Start: {best_segment['start']}, "
      f"Duration: {best_segment['duration']}, Similarity: {best_segment['similarity']}")
