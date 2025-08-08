import json
import re

import torch
from sentence_transformers import SentenceTransformer

from improved_preprocessing import preprocess_text


class ProductSemanticSearch:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        """
        다국어 지원 모델을 초기화합니다.
        paraphrase-multilingual-MiniLM-L12-v2는 한국어와 영어 모두 지원합니다.
        """
        self.model = SentenceTransformer(model_name)

        # 제품 관련 키워드 사전 정의
        self.product_related_terms = [
            # 제품 관련 일반 용어
            "제품", "상품", "물건", "품목", "아이템", "제조", "생산",
            "브랜드", "메이커", "판매", "구매", "구입", "사용", "소비",
            "가격", "비용", "원가", "할인", "프로모션", "세일", "품질",
            "기능", "성능", "스펙", "디자인", "모델", "버전", "출시",
            "신제품", "최신", "인기", "추천", "리뷰", "평가", "사용법",

            # 전자제품 관련 용어
            "스마트폰", "핸드폰", "태블릿", "노트북", "컴퓨터", "pc", "모니터",
            "tv", "티비", "냉장고", "세탁기", "에어컨", "청소기", "카메라",
            "이어폰", "헤드폰", "스피커", "프린터", "충전기", "배터리", "스팀청소기"

            # 화장품/뷰티 관련 용어
                                                      "화장품", "스킨케어", "메이크업", "립스틱", "파운데이션", "마스카라",
            "팩트", "쿠션", "선크림", "토너", "로션", "에센스", "크림", "세럼",
            "클렌징", "샴푸", "컨디셔너", "향수", "마스크팩",

            # 식품 관련 용어
            "음식", "식품", "간식", "음료", "식재료", "먹거리", "요리",

            # 의류/패션 관련 용어
            "옷", "의류", "패션", "신발", "가방", "액세서리", "모자", "장갑",
            "양말", "스카프", "벨트", "시계", "쥬얼리",

            # 아기 용품 관련 용어
            "기저귀", "분유", "젖병", "유모차", "카시트", "아기띠", "보호대",
            "이유식", "장난감", "아기옷", "내의", "바디슈트", "우유", "분유포트",
            "젖병소독기", "아기침대", "범퍼침대", "아기욕조", "샴푸", "로션",
            "물티슈", "딸랑이", "치발기", "모빌", "젖꼭지", "노리개", "아기과자",
            "이유식용품", "아기한테", "아이한테", "애기", "유아", "아기용", "신생아",
            "아동복", "돌선물", "수면조끼", "속싸개", "겉싸개", "아기침구", "살균기"
        ]

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
            text = subtitle.get("text", "")
            processed_text = preprocess_text(text)

            # 제품 관련 키워드 포함 여부 확인
            product_keyword_count = self._count_product_keywords(processed_text)

            self.subtitle_data.append({
                "text": text,
                "processed_text": processed_text,
                "start": subtitle.get("start", 0),
                "duration": subtitle.get("duration", 0),
                "embedding": embeddings[i],
                "product_keyword_count": product_keyword_count
            })

        return self.subtitle_data

    def _count_product_keywords(self, text):
        """
        텍스트에 제품 관련 키워드가 몇 개 포함되어 있는지 계산합니다.
        """
        count = 0
        for term in self.product_related_terms:
            # 단어 경계를 고려하여 매칭 (정확한 단어 매칭)
            pattern = r'\b' + re.escape(term) + r'\b'
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count

    def search_product_keyword(self, keyword, top_k=5, product_weight=0.3):
        """
        키워드와 의미적으로 유사하면서 제품 관련 키워드를 많이 포함한 자막 구간을 찾습니다.

        product_weight: 제품 관련 키워드 가중치 (0~1 사이 값).
                       0이면 순수 시맨틱 검색, 1이면 제품 키워드 기반 검색
        """
        if not hasattr(self, 'subtitle_data'):
            raise ValueError("먼저 create_embeddings 메서드를 호출하여 임베딩을 생성해야 합니다.")

        # 키워드 임베딩 생성
        keyword_embedding = self.model.encode(keyword, convert_to_tensor=True)

        # 제품 관련 키워드 수 최대값 계산 (정규화를 위해)
        max_product_count = max(
            [subtitle["product_keyword_count"] for subtitle in self.subtitle_data]) if self.subtitle_data else 1

        # 모든 자막 임베딩과의 코사인 유사도 계산 + 제품 키워드 가중치 적용
        similarities = []
        for subtitle in self.subtitle_data:
            # 시맨틱 유사도 계산
            semantic_similarity = torch.nn.functional.cosine_similarity(
                keyword_embedding.unsqueeze(0),
                subtitle["embedding"].unsqueeze(0)
            ).item()

            # 제품 키워드 점수 계산 (0~1 사이로 정규화)
            product_score = subtitle["product_keyword_count"] / max_product_count if max_product_count > 0 else 0

            # 최종 유사도 점수 계산 (가중 평균)
            final_similarity = (1 - product_weight) * semantic_similarity + product_weight * product_score

            similarities.append({
                "text": subtitle["text"],
                "processed_text": subtitle["processed_text"],
                "start": subtitle["start"],
                "duration": subtitle["duration"],
                "semantic_similarity": semantic_similarity,
                "product_score": product_score,
                "final_similarity": final_similarity,
                "product_keyword_count": subtitle["product_keyword_count"]
            })

        # 최종 유사도 기준으로 내림차순 정렬
        sorted_results = sorted(similarities, key=lambda x: x["final_similarity"], reverse=True)

        # 상위 k개 결과 반환
        return sorted_results[:top_k]

    def search_with_context(self, keyword, top_k=5, context_window=1, product_weight=0.3):
        """
        키워드와 유사한 구간을 찾고, 전후 맥락도 함께 반환합니다.
        """
        results = self.search_product_keyword(keyword, top_k, product_weight)

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
    searcher = ProductSemanticSearch()
    json_data = searcher.load_subtitles("gKEzL3pn1VA.json")
    searcher.create_embeddings(json_data)

    # 사용자 키워드 입력 받기
    keyword = input("검색할 제품 키워드를 입력하세요: ")
    product_weight = float(input("제품 키워드 가중치를 입력하세요 (0.0 ~ 1.0): "))

    # 유사한 구간 찾기
    results = searcher.search_product_keyword(keyword, top_k=5, product_weight=product_weight)

    print(f"\n'{keyword}'와 의미적으로 유사한 상위 5개 구간:")
    for i, result in enumerate(results, 1):
        print(f"{i}. 시작: {result['start']:.2f}초, 최종 유사도: {result['final_similarity']:.4f}")
        print(f"   시맨틱 유사도: {result['semantic_similarity']:.4f}, 제품 점수: {result['product_score']:.4f}")
        print(f"   제품 관련 키워드 수: {result['product_keyword_count']}")
        print(f"   원본 텍스트: {result['text']}")
        print(f"   전처리 텍스트: {result['processed_text']}")
        print()

    # 컨텍스트와 함께 결과 확인하기
    print("\n컨텍스트와 함께 보기:")
    context_results = searcher.search_with_context(keyword, top_k=3, context_window=2, product_weight=product_weight)

    for i, context in enumerate(context_results, 1):
        print(f"\n{i}. 주요 구간 (최종 유사도: {context['main']['final_similarity']:.4f}):")
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
