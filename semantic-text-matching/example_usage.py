from semantic_search import SemanticSubtitleSearch


def main():
    # 자막 파일 로드
    file_name = "gWirXv763N4.json"  # 예시 파일 이름

    # 1. 시맨틱 검색을 위한 검색기 초기화
    searcher = SemanticSubtitleSearch()

    # 2. 자막 데이터 로드
    json_data = searcher.load_subtitles(file_name)

    # 3. 임베딩 생성
    searcher.create_embeddings(json_data)

    # 4. 검색 수행
    print("=" * 50)
    print("유튜브 자막 시맨틱 검색 도구")
    print("=" * 50)

    while True:
        # 사용자 키워드 입력 받기
        keyword = input("\n검색할 제품 키워드를 입력하세요 (종료하려면 'q' 입력): ")

        if keyword.lower() == 'q':
            break

        # 유사한 구간 찾기
        try:
            results = searcher.search_keyword(keyword, top_k=5)

            if not results:
                print(f"'{keyword}'와 관련된 구간을 찾을 수 없습니다.")
                continue

            print(f"\n'{keyword}'와 의미적으로 유사한 상위 5개 구간:")
            for i, result in enumerate(results, 1):
                print(f"{i}. 시작: {result['start']:.2f}초, 유사도: {result['similarity']:.4f}")
                print(f"   원본 텍스트: {result['text']}")
                print(f"   전처리 텍스트: {result['processed_text']}")
                print()

            # 컨텍스트 보기 옵션
            show_context = input("전후 맥락을 함께 보시겠습니까? (y/n): ")
            if show_context.lower() == 'y':
                context_window = 2  # 전후 각각 2개 자막 보기
                context_results = searcher.search_with_context(keyword, top_k=3, context_window=context_window)

                print("\n컨텍스트와 함께 보기:")
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

        except Exception as e:
            print(f"검색 중 오류가 발생했습니다: {e}")

    print("\n프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
