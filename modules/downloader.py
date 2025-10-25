import os
import subprocess
import yt_dlp


def download_youtube_video(
    youtube_url: str,
    raw_video_output_dir: str = "raw_videos",
    quality: int = 1080,
) -> str:

    os.makedirs(raw_video_output_dir, exist_ok=True)

    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: FFmpeg is not installed or not found in your system's PATH.")
        print("FFmpeg is required to merge high-quality video and audio streams from YouTube.")
        print("Please install FFmpeg and ensure it's accessible from your command line.")
        return ""

    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            video_title = info_dict.get('title', 'youtube_video')
            # Sanitize the title to make it a valid filename
            sanitized_title = "".join([c for c in video_title if c.isalpha() or c.isdigit() or c in (' ', '-')]).rstrip()
            output_filename = f"{sanitized_title}.mp4"
    except Exception as e:
        print(f"Error fetching video title: {e}")
        # Fallback to a generic name if title fetch fails
        output_filename = "downloaded_video.mp4"

    output_path = os.path.join(raw_video_output_dir, output_filename)

    ydl_opts = {
        'format': f'bv*[height<={quality}]+ba/b[height<={quality}]',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
        'quiet': False,
        'progress': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
    }

    print(f"Starting download for: {youtube_url}")
    print(f"Requested quality: {quality}p")
    print(f"Saving to: {output_path}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        if os.path.exists(output_path):
            print("\nDownload and processing complete.")
            print(f"Video saved successfully at: {output_path}")
            return output_path
        else:
            print("\nDownload failed. The output file was not created.")
            return ""

    except Exception as e:
        print(f"\nAn error occurred during the download process: {e}")
        return ""


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Download a YouTube video to MP4")
    ap.add_argument("--url", required=True, help="YouTube URL")
    ap.add_argument("--quality", type=int, default=1080, help="Max height")
    ap.add_argument("--out", default="raw_videos", help="Output folder")
    args = ap.parse_args()
    path = download_youtube_video(args.url, args.out, args.quality)
    print(path)
