# F-001: 데이터 파이프라인 — JSON → Parquet 변환 + 로더

Issue: #1

## 개요

100만 건 JSON을 매번 로드하는 비효율을 해결하기 위해 Parquet 변환 스크립트와 날짜 범위 기반 로더를 구현한다.

## 아키텍처

```
user-churn-py/
  ├── convert_to_parquet.py   ← JSON → Parquet 변환 (1회성)
  ├── data_loader.py          ← load_parquet() 추가
  └── data/matches/           ← Parquet 파티셔닝 디렉토리
        ├── 2015-05-25.parquet
        ├── 2015-05-26.parquet
        └── ...
```

## 시나리오 명세

### SC-1: JSON → Parquet 변환 스크립트

**Given** `bulkmatches1-20000.json` 파일이 존재할 때
**When** `python convert_to_parquet.py`를 실행하면
**Then** `data/matches/` 디렉토리에 날짜별 Parquet 파일이 생성된다

구현 요구:
- 기존 `load_df()` 로직을 재활용하여 DataFrame 생성
- `date` 컬럼의 날짜(yyyy-mm-dd)별로 파티셔닝하여 저장
- 이미 변환된 경우 스킵 옵션 제공 (`--force` 플래그)
- 변환 완료 후 파일 수, 총 레코드 수 출력

### SC-2: Parquet 기반 데이터 로더

**Given** `data/matches/` 디렉토리에 Parquet 파일이 존재할 때
**When** 날짜 범위를 지정하여 `load_parquet(start_date, end_date)`를 호출하면
**Then** 해당 범위의 데이터만 로드하여 DataFrame을 반환한다

구현 요구:
- `data_loader.py`에 `load_parquet(data_dir, start_date, end_date)` 함수 추가
- 날짜 범위에 해당하는 Parquet 파일만 선택적으로 로드
- 빈 범위일 경우 빈 DataFrame 반환 (에러 아님)
- 기존 `load_df()`, `filter_df()`, `data_split()`은 그대로 유지
- 반환 DataFrame 스키마는 기존 `load_df()`와 동일

## 추적성 매트릭스

| AC | SC | 설명 |
|----|-----|------|
| AC-1 | SC-1 | JSON → Parquet 변환 후 날짜별 파일 생성 |
| AC-2 | SC-2 | 날짜 범위 지정 시 해당 범위만 로드 |
| AC-3 | SC-2 | 빈 범위 시 빈 DataFrame 반환 |

## 기술 결정

- **DEC-A1**: Parquet (날짜별 파티셔닝) — 컬럼 압축으로 용량 1/5~1/10, 날짜 필터 시 필요한 파일만 로드
- **DEC-F1**: 변환 스크립트 분리 — 변환과 분석의 관심사 분리
