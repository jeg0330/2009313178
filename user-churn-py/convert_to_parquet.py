"""JSON -> Parquet 변환 스크립트.

data/origin/ 디렉토리의 JSON 파일들을 읽어 날짜별 Parquet 파일로 변환한다.
출력 경로: data/matches/<yyyy-mm-dd>.parquet
"""

import argparse
import sys
from pathlib import Path

from data_loader import load_df


ORIGIN_DIR = Path("data/origin")
OUTPUT_DIR = Path("data/matches")


def convert(force: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_files = set(OUTPUT_DIR.glob("*.parquet"))
    if existing_files and not force:
        print(
            f"[SKIP] {len(existing_files)}개의 Parquet 파일이 이미 존재합니다. "
            "덮어쓰려면 --force 플래그를 사용하세요."
        )
        sys.exit(0)

    # data/origin/ 디렉토리의 모든 JSON 파일을 찾아 로드
    json_files = sorted(ORIGIN_DIR.glob("*.json"))
    if not json_files:
        print(f"[ERROR] {ORIGIN_DIR}에 JSON 파일이 없습니다.")
        sys.exit(1)

    print(f"{len(json_files)}개 JSON 파일 발견: {[f.name for f in json_files]}")

    import pandas as pd
    dfs = []
    for json_file in json_files:
        print(f"\n--- {json_file.name} 로드 중 ---")
        df = load_df(str(json_file))
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n전체 레코드: {len(combined)}건")

    # 날짜별 파티셔닝
    combined["date_str"] = combined["date"].dt.strftime("%Y-%m-%d")

    file_count = 0
    total_records = 0

    for date_str, group in combined.groupby("date_str"):
        out_path = OUTPUT_DIR / f"{date_str}.parquet"
        group.drop(columns=["date_str"]).to_parquet(out_path, index=False)
        file_count += 1
        total_records += len(group)

    print(f"\n변환 완료: {file_count}개 파일, 총 {total_records}건")


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
