import json

from product_semantic_search import ProductSemanticSearch


def main():
    # 자막 파일 로드
    file_name = "gWirXv763N4.json"  # 예시 파일 이름

    # 1. 제품 시맨틱 검색을 위한 검색기 초기화
    searcher = ProductSemanticSearch()

    # 2. 자막 데이터 로드
    try:
        json_data = searcher.load_subtitles(file_name)
    except FileNotFoundError:
        print(f"오류: '{file_name}' 파일을 찾을 수 없습니다.")
        print("먼저 fetch_subtitle.py를 실행하여 자막을 다운로드해주세요.")
        return
    except json.JSONDecodeError:
        print(f"오류: '{file_name}' 파일이 올바른 JSON 형식이 아닙니다.")
        return

    # 3. 임베딩 생성
    print("자막 임베딩을 생성 중입니다...")
    searcher.create_embeddings(json_data)
    print("임베딩 생성이 완료되었습니다.")

    # 4. 검색 수행
    print("\n" + "=" * 50)
    print("유튜브 자막 제품 시맨틱 검색 도구")
    print("=" * 50)
    print("\n이 도구는 제품 관련 키워드를 유튜브 자막에서 검색합니다.")
    print("제품 키워드 가중치를 조절하여 검색 결과를 최적화할 수 있습니다.")
    print("- 가중치 0.0: 순수한 의미적(시맨틱) 검색")
    print("- 가중치 1.0: 제품 관련 키워드 중심 검색")
    print("- 권장 가중치: 0.3 ~ 0.5")

    # 제품 가중치 기본값 설정
    default_weight = 0.3

    while True:
        # 사용자 키워드 입력 받기
        keyword = input("\n검색할 제품 키워드를 입력하세요 (종료하려면 'q' 입력): ")

        if keyword.lower() == 'q':
            break

        # 제품 가중치 입력
        weight_input = input(f"제품 키워드 가중치를 입력하세요 (0.0 ~ 1.0, 기본값: {default_weight}): ")
        if weight_input.strip() == "":
            product_weight = default_weight
        else:
            try:
                product_weight = float(weight_input)
                if product_weight < 0 or product_weight > 1:
                    print("가중치는 0.0에서 1.0 사이의 값이어야 합니다. 기본값을 사용합니다.")
                    product_weight = default_weight
            except ValueError:
                print("올바른 숫자 형식이 아닙니다. 기본값을 사용합니다.")
                product_weight = default_weight

        # 유사한 구간 찾기
        try:
            results = searcher.search_product_keyword(keyword, top_k=5, product_weight=product_weight)

            if not results:
                print(f"'{keyword}'와 관련된 구간을 찾을 수 없습니다.")
                continue

            print(f"\n'{keyword}'와 관련된 상위 5개 구간 (가중치: {product_weight}):")
            for i, result in enumerate(results, 1):
                print(f"{i}. 시작: {result['start']:.2f}초, 최종 유사도: {result['final_similarity']:.4f}")
                print(f"   시맨틱 유사도: {result['semantic_similarity']:.4f}, 제품 점수: {result['product_score']:.4f}")
                if result['product_keyword_count'] > 0:
                    print(f"   제품 관련 키워드 수: {result['product_keyword_count']}")
                print(f"   원본 텍스트: {result['text']}")
                print()

            # 컨텍스트 보기 옵션
            show_context = input("전후 맥락을 함께 보시겠습니까? (y/n): ")
            if show_context.lower() == 'y':
                context_window_input = input("맥락 창 크기를 입력하세요 (기본값: 2): ")
                try:
                    context_window = int(context_window_input) if context_window_input.strip() else 2
                except ValueError:
                    context_window = 2
                    print("올바른 숫자 형식이 아닙니다. 기본값 2를 사용합니다.")

                context_results = searcher.search_with_context(
                    keyword, top_k=3, context_window=context_window, product_weight=product_weight
                )

                print("\n컨텍스트와 함께 보기:")
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

        except Exception as e:
            print(f"검색 중 오류가 발생했습니다: {e}")

    print("\n프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
