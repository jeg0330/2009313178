from youtube_transcript_api import YouTubeTranscriptApi
import json

def fetch_youtube_subtitles(video_id):
    try:
        # 한국어와 영어 자막을 우선 조회합니다.
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
        return transcript
    except Exception as e:
        print("자막을 가져오는 중 오류 발생:", e)
        return None

def save_subtitles_to_file(video_id, subtitles):
    # 파일 이름을 video_id.json 형태로 지정
    file_name = f"{video_id}.json"
    data_to_save = {
        "video_id": video_id,
        "subtitles": subtitles
    }
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data_to_save, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    video_id = "gKEzL3pn1VA"  # 실제 유튜브 video id 입력
    subtitles = fetch_youtube_subtitles(video_id)
    if subtitles:
        save_subtitles_to_file(video_id, subtitles)
        print(f"자막이 {video_id}.json 파일로 저장되었습니다.")
    else:
        print("자막을 가져오지 못했습니다.")