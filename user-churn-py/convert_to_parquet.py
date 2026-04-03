"""JSON -> Parquet 변환 스크립트.

bulkmatches1-20000.json 파일을 읽어 날짜별 Parquet 파일로 변환한다.
출력 경로: data/matches/<yyyy-mm-dd>.parquet
"""

import argparse
import sys
from pathlib import Path

from data_loader import load_df


DATA_DIR = Path("data/matches")
JSON_FILE = "bulkmatches1-20000.json"


def convert(force: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing_files = set(DATA_DIR.glob("*.parquet"))
    if existing_files and not force:
        print(
            f"[SKIP] {len(existing_files)}개의 Parquet 파일이 이미 존재합니다. "
            "덮어쓰려면 --force 플래그를 사용하세요."
        )
        sys.exit(0)

    # 기존 load_df() 로직을 재활용하여 DataFrame 생성
    df = load_df(JSON_FILE)

    # 날짜 문자열 컬럼 생성 (yyyy-mm-dd)
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    file_count = 0
    total_records = 0

    for date_str, group in df.groupby("date_str"):
        out_path = DATA_DIR / f"{date_str}.parquet"
        group.drop(columns=["date_str"]).to_parquet(out_path, index=False)
        file_count += 1
        total_records += len(group)

    print(f"변환 완료: {file_count}개 파일, 총 {total_records}건")


def main() -> None:
    parser = argparse.ArgumentParser(description="JSON -> Parquet 변환")
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 Parquet 파일을 덮어쓰기",
    )
    args = parser.parse_args()
    convert(force=args.force)


if __name__ == "__main__":
    main()
