import json

import torch
from sentence_transformers import SentenceTransformer

from preprocessing import preprocess_text


class SemanticSubtitleSearch:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        """
        다국어 지원 모델을 초기화합니다.
        paraphrase-multilingual-MiniLM-L12-v2는 한국어와 영어 모두 지원합니다.
        """
        self.model = SentenceTransformer(model_name)

    def load_subtitles(self, json_file_path):
        """
        자막 JSON 파일을 로드합니다.
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        return json_data

    def create_embeddings(self, json_data):
        """
        자막 데이터의 임베딩을 생성합니다.
        """
        subtitles = json_data.get("subtitles", [])
        texts = [subtitle.get("text", "") for subtitle in subtitles]

        # 각 자막 텍스트에 대한 임베딩 생성
        embeddings = self.model.encode(texts, convert_to_tensor=True)

        # 임베딩과 함께 원본 자막 정보 저장
        self.subtitle_data = []
        for i, subtitle in enumerate(subtitles):
            self.subtitle_data.append({
                "text": subtitle.get("text", ""),
                "processed_text": preprocess_text(subtitle.get("text", "")),
                "start": subtitle.get("start", 0),
                "duration": subtitle.get("duration", 0),
                "embedding": embeddings[i]
            })

        return self.subtitle_data

    def search_keyword(self, keyword, top_k=5):
        """
        키워드와 의미적으로 유사한 자막 구간을 찾습니다.
        """
        if not hasattr(self, 'subtitle_data'):
            raise ValueError("먼저 create_embeddings 메서드를 호출하여 임베딩을 생성해야 합니다.")

        # 키워드 임베딩 생성
        keyword_embedding = self.model.encode(keyword, convert_to_tensor=True)

        # 모든 자막 임베딩과의 코사인 유사도 계산
        similarities = []
        for subtitle in self.subtitle_data:
            similarity = torch.nn.functional.cosine_similarity(
                keyword_embedding.unsqueeze(0),
                subtitle["embedding"].unsqueeze(0)
            ).item()
            similarities.append({
                "text": subtitle["text"],
                "processed_text": subtitle["processed_text"],
                "start": subtitle["start"],
                "duration": subtitle["duration"],
                "similarity": similarity
            })

        # 유사도 기준으로 내림차순 정렬
        sorted_results = sorted(similarities, key=lambda x: x["similarity"], reverse=True)

        # 상위 k개 결과 반환
        return sorted_results[:top_k]

    def search_with_context(self, keyword, top_k=5, context_window=1):
        """
        키워드와 유사한 구간을 찾고, 전후 맥락도 함께 반환합니다.
        """
        results = self.search_keyword(keyword, top_k)

        # 각 결과에 대해 전후 맥락 추가
        contextualized_results = []

        for result in results:
            # 현재 자막의 인덱스 찾기
            for i, subtitle in enumerate(self.subtitle_data):
                if subtitle["start"] == result["start"] and subtitle["text"] == result["text"]:
                    current_index = i
                    break

            # 전후 컨텍스트 추가
            context = {
                "main": result,
                "before": [],
                "after": []
            }

            # 이전 컨텍스트 추가
            for j in range(max(0, current_index - context_window), current_index):
                context["before"].append({
                    "text": self.subtitle_data[j]["text"],
                    "start": self.subtitle_data[j]["start"],
                    "duration": self.subtitle_data[j]["duration"]
                })

            # 이후 컨텍스트 추가
            for j in range(current_index + 1, min(len(self.subtitle_data), current_index + context_window + 1)):
                context["after"].append({
                    "text": self.subtitle_data[j]["text"],
                    "start": self.subtitle_data[j]["start"],
                    "duration": self.subtitle_data[j]["duration"]
                })

            contextualized_results.append(context)

        return contextualized_results


def main():
    # 예제 사용법
    searcher = SemanticSubtitleSearch()
    json_data = searcher.load_subtitles("gKEzL3pn1VA.json")
    searcher.create_embeddings(json_data)

    # 사용자 키워드 입력 받기
    keyword = input("검색할 제품 키워드를 입력하세요: ")

    # 유사한 구간 찾기
    results = searcher.search_keyword(keyword, top_k=5)

    print(f"\n'{keyword}'와 의미적으로 유사한 상위 5개 구간:")
    for i, result in enumerate(results, 1):
        print(f"{i}. 시작: {result['start']:.2f}초, 유사도: {result['similarity']:.4f}")
        print(f"   원본 텍스트: {result['text']}")
        print(f"   전처리 텍스트: {result['processed_text']}")
        print()

    # 컨텍스트와 함께 결과 확인하기
    print("\n컨텍스트와 함께 보기:")
    context_results = searcher.search_with_context(keyword, top_k=3, context_window=2)

    for i, context in enumerate(context_results, 1):
        print(f"\n{i}. 주요 구간 (유사도: {context['main']['similarity']:.4f}):")
        print(f"   시작: {context['main']['start']:.2f}초, 텍스트: {context['main']['text']}")

        if context['before']:
            print("\n   이전 컨텍스트:")
            for before in context['before']:
                print(f"   - {before['start']:.2f}초: {before['text']}")

        if context['after']:
            print("\n   이후 컨텍스트:")
            for after in context['after']:
                print(f"   - {after['start']:.2f}초: {after['text']}")

        print("\n" + "-" * 50)


if __name__ == "__main__":
    main()
