# Data - 샘플 자막 데이터

테스트와 개발에 사용할 수 있는 유튜브 자막 샘플 데이터입니다.

## 파일 목록
- `gKEzL3pn1VA.json` - 아기용품 관련 영상 자막
- `gWirXv763N4.json` - 샘플 영상 자막 
- `M2y2wWAYXNU.json` - 샘플 영상 자막

## 파일 구조
```json
{
    "video_id": "gKEzL3pn1VA",
    "subtitles": [
        {
            "text": "자막 텍스트",
            "start": 2.12,      // 시작 시간(초)
            "duration": 6.479   // 지속 시간(초)
        }
    ]
}
```

## 사용법
```python
import json

with open('data/gKEzL3pn1VA.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
subtitles = data['subtitles']
```

## 테스트 키워드 예시
- **아기용품**: "맘마", "젖병", "아기", "분유"
- **일반**: 영상 내용에 따라 다름