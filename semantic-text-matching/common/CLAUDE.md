# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Korean/multilingual semantic text matching system for YouTube subtitle search. The system extracts YouTube video subtitles and performs semantic similarity searches to find relevant content segments based on keywords, with specialized product search capabilities.

## Core Architecture

### Main Components

- **SemanticSubtitleSearch** (`semantic_search.py`): Core semantic search engine using multilingual sentence transformers
- **ProductSemanticSearch** (`product_semantic_search.py`): Specialized search for product-related content with keyword weighting
- **Text Preprocessing Pipeline**: Multiple preprocessing modules for Korean/English text normalization
- **Subtitle Data Pipeline**: YouTube subtitle extraction, processing, and embedding generation

### Processing Flow

1. **Data Acquisition**: `fetch_subtitle.py` → YouTube Transcript API → JSON storage
2. **Preprocessing**: Text normalization, sentence segmentation, contiguous segment grouping
3. **Embedding Generation**: Sentence-transformers multilingual models (all-MiniLM-L6-v2, paraphrase-multilingual-MiniLM-L12-v2)
4. **Semantic Search**: Cosine similarity calculation with optional product keyword weighting
5. **Context Extraction**: Surrounding subtitle segments for comprehensive results

### Key Design Patterns

- **Modular Processing**: Separate modules for embedding, preprocessing, similarity, and segment extraction
- **Metadata Preservation**: Original subtitle timing and text preserved throughout pipeline
- **Hybrid Scoring**: Combines semantic similarity with domain-specific keyword matching
- **Context-Aware Results**: Returns surrounding subtitle segments for better understanding

## Development Commands

### Running the Application

```bash
# Basic semantic search (interactive)
python semantic_search.py

# Product-focused search (interactive)
python enhanced_product_search.py

# Example usage demonstration
python example_usage.py

# Legacy pipeline demonstration
python main.py
```

### Data Acquisition

```bash
# Fetch subtitles for a specific video
# Edit video_id in fetch_subtitle.py before running
python fetch_subtitle.py
```

### Dependencies

The project requires these key libraries:
- `sentence-transformers` - Multilingual semantic embeddings
- `youtube-transcript-api` - YouTube subtitle extraction
- `torch` - PyTorch backend for transformers
- `sklearn` - Cosine similarity calculations
- `numpy` - Numerical operations

## File Structure and Responsibilities

### Core Search Engines
- `semantic_search.py`: Main semantic search with context support
- `product_semantic_search.py`: Product-specialized search with keyword weighting

### Text Processing
- `preprocessing.py`: Basic text cleaning and sentence segmentation
- `improved_preprocessing.py`: Enhanced normalization with Unicode handling
- `segment_extraction.py`: Best segment identification with temporal grouping

### Utilities
- `embedding.py`: Simple embedding generation wrapper
- `similarity.py`: Cosine similarity calculation
- `fetch_subtitle.py`: YouTube subtitle extraction and JSON storage

### Data Files
- `*.json`: YouTube subtitle data with video_id, text, start, duration structure

## Configuration and Models

### Model Selection
- Default: `paraphrase-multilingual-MiniLM-L12-v2` for production search
- Alternative: `all-MiniLM-L6-v2` for basic embedding generation
- Both models support Korean and English text

### Product Search Configuration
- Product keyword dictionary in `product_semantic_search.py` includes electronics, cosmetics, food, clothing, baby products
- Adjustable product weighting (0.0 = pure semantic, 1.0 = keyword-focused)
- Default product weight: 0.3

### Text Processing Parameters
- `max_gap`: Maximum time gap for segment grouping (default: 1.0 seconds)
- `similarity_threshold`: Minimum similarity for adjacent segment inclusion (default: 0.5)
- `context_window`: Number of surrounding segments to include (default: 2)

## Key Implementation Details

### Subtitle Data Structure
```python
{
    "video_id": "string",
    "subtitles": [
        {
            "text": "subtitle text",
            "start": float,  # seconds
            "duration": float  # seconds
        }
    ]
}
```

### Search Result Structure
```python
{
    "text": "original subtitle text",
    "processed_text": "cleaned text",
    "start": float,
    "duration": float,
    "similarity": float,  # or final_similarity for product search
    "semantic_similarity": float,  # product search only
    "product_score": float,  # product search only
    "product_keyword_count": int  # product search only
}
```

### Context Result Structure
```python
{
    "main": result_object,
    "before": [subtitle_objects],
    "after": [subtitle_objects]
}
```

## Common Development Patterns

- **Interactive CLI**: Most main functions provide interactive keyword input loops
- **Error Handling**: JSON decoding errors, file not found, and API exceptions handled gracefully
- **Bilingual Support**: Korean/English processing with Unicode normalization
- **Temporal Awareness**: All operations preserve and utilize subtitle timing information
- **Extensible Keyword Matching**: Product keywords easily expandable in dedicated dictionary

## Testing and Validation

Sample data files (`gKEzL3pn1VA.json`, `M2y2wWAYXNU.json`, `gWirXv763N4.json`) are included for development and testing. These contain real YouTube subtitle data with Korean text for validation.

When making changes to text preprocessing or search algorithms, test with these sample files to ensure Korean text handling remains correct.