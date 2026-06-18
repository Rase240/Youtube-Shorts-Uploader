import argparse
import asyncio
import requests
import uuid
import os
from scheduler import Job, run_batch

async def main():
    parser = argparse.ArgumentParser(description="CLI for uploading a single video to YouTube via youtube_bot")
    parser.add_argument('--vibe', required=True, help="The vibe of the video")
    parser.add_argument('--drive_url', required=False, help="Google Drive link")
    parser.add_argument('--discord_url', required=False, help="Direct Discord attachment link")
    parser.add_argument('--genre', default='comedy', help="Genre for YouTube category")
    parser.add_argument('--privacy', default='public', help="Privacy status (public, private, unlisted)")

    args = parser.parse_args()

    video_path = None
    if args.discord_url:
        os.makedirs("videos", exist_ok=True)
        video_path = f"videos/temp_discord_{uuid.uuid4().hex[:8]}.mp4"
        print(f"Downloading from Discord: {args.discord_url}")
        try:
            r = requests.get(args.discord_url, stream=True)
            r.raise_for_status()
            with open(video_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded to {video_path}")
        except Exception as e:
            print(f"Failed to download from Discord: {e}")
            return

    job = Job(
        vibe=args.vibe,
        drive_url=args.drive_url,
        video_path=video_path,
        genre=args.genre,
        default_privacy=args.privacy
    )
    
    try:
        video_ids = await run_batch([job], max_concurrent=1)
        if video_ids:
            print(f"SUCCESS: {video_ids[0]}")
        else:
            print("FAILED")
    except Exception as e:
        print(f"Job Failed: {e}")
    finally:
        if args.discord_url and video_path and os.path.exists(video_path):
            os.remove(video_path)
            print(f"Cleaned up temp file {video_path}")

if __name__ == '__main__':
    asyncio.run(main())
