# Common - 공통 유틸리티

모든 버전에서 공통으로 사용하는 유틸리티와 문서입니다.

## 파일 구성
- `fetch_subtitle.py` - 유튜브 자막 다운로드
- `CLAUDE.md` - 프로젝트 전체 가이드

## 자막 다운로드 사용법

### 1. 스크립트 직접 실행
```bash
cd common
python fetch_subtitle.py
```

### 2. 다른 스크립트에서 사용
```python
from fetch_subtitle import fetch_youtube_subtitles, save_subtitles_to_file

# 자막 다운로드
video_id = "gKEzL3pn1VA"  
subtitles = fetch_youtube_subtitles(video_id)
save_subtitles_to_file(video_id, subtitles)
```

## 의존성
- `youtube-transcript-api` - 유튜브 자막 API

## 지원 언어
- 한국어 (ko)
- 영어 (en)