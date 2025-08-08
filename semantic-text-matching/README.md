# 유튜브 자막 시맨틱 검색 시스템

이전에 본 유튜브 영상에서 특정 제품이나 음식점이 언급된 구간을 키워드로 빠르게 찾아주는 시스템입니다.

## 🎯 프로젝트 목적
- 유튜브 영상에서 제품/음식점 언급 구간을 빠르게 찾기
- 슬라이드를 넘기며 찾는 번거로움 해결
- 정확한 타임라인 정보로 해당 구간 바로 이동

## 📁 프로젝트 구조

```
📦 semantic-text-matching
├── 📁 v1/                 # 기본 버전 (초기 구현)
│   ├── main.py
│   ├── embedding.py
│   ├── preprocessing.py
│   └── README.md
├── 📁 v2/                 # 제품 특화 버전
│   ├── enhanced_product_search.py
│   ├── semantic_search.py
│   ├── product_semantic_search.py
│   └── README.md
├── 📁 v3/                 # 차세대 하이브리드 버전 ⭐
│   ├── enhanced_semantic_search.py  # 최고 정확도
│   ├── improved_preprocessing.py
│   └── README.md
├── 📁 common/             # 공통 유틸리티
│   ├── fetch_subtitle.py  # 자막 다운로드
│   └── CLAUDE.md          # 개발 가이드
└── 📁 data/               # 샘플 자막 데이터
    ├── gKEzL3pn1VA.json
    └── README.md
```

## 🚀 빠른 시작

### 1. 의존성 설치
```bash
pip install sentence-transformers youtube-transcript-api torch scikit-learn
```

### 2. 자막 다운로드
```bash
cd common
python fetch_subtitle.py
```

### 3. 검색 실행 (추천: V3)
```bash
cd v3
python enhanced_semantic_search.py
```

## ✨ 버전별 특징

### V1 - 기본 버전
- 단순한 시맨틱 검색
- 모듈형 구조
- 낮은 검색 정확도

### V2 - 제품 특화 버전
- **제품 키워드 사전** - 아기용품, 전자제품, 화장품 등
- **가중치 조절 기능** - 사용자가 비중 조절 가능
- **개선된 UI** - 사용자 친화적 인터페이스

### V3 - 차세대 하이브리드 버전 ⭐
- **직접 키워드 매칭** - 정확한 단어 우선순위
- **하이브리드 검색** - 직접매칭(40%) + 시맨틱(30%) + 제품키워드(30%)
- **지능형 매칭** - 띄어쓰기 변형, 부분 매칭 지원
- **최고 검색 정확도**

## 📊 성능 개선 결과

### Before (V1)
```
"맘마" 검색 → 실제 "맘마 존" 구간 찾지 못함 ❌
```

### After (V3)
```
"맘마" 검색 결과:
1. 시작: 597.6초 - 응 여기가 저희 맘마 존이 완성됐어요
2. 시작: 8.6초 - 여기가 저희 맘마 존이 완성됐어요 ✅
```

## 🎮 사용 예시

### 검색 결과
```
키워드: "아기"
결과:
1. 시작: 640.2초, 점수: 0.9165
   텍스트: 아기가 급하게 추가로 우유 먹고 싶어
   
2. 시작: 737.1초, 점수: 0.6542  
   텍스트: 나와요 응 50 나와 그니까 아기가
```

### 유튜브 링크 생성
결과의 시작 시간으로 바로 해당 구간 이동:
`https://youtu.be/gKEzL3pn1VA?t=640`

## 🔧 개발자 가이드

각 폴더의 `README.md`와 `common/CLAUDE.md`를 참고하세요.

## 🎯 권장 사용법

1. **최고 정확도**: `v3/enhanced_semantic_search.py` ⭐
2. **제품 위주 검색**: `v2/enhanced_product_search.py`
3. **기본 사용**: `v1/main.py`
4. **개발자**: 각 버전 폴더의 README 참고