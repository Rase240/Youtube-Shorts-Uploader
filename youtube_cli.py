import argparse
import asyncio
import requests
import uuid
import os
import json
import sys
from scheduler import Job, run_batch, process_job

async def handle_upload(args):
    video_path = None
    if args.discord_url:
        os.makedirs("videos", exist_ok=True)
        video_path = f"videos/temp_discord_{uuid.uuid4().hex[:8]}.mp4"
        print(f"Downloading from Discord: {args.discord_url}", file=sys.stderr)
        try:
            r = requests.get(args.discord_url, stream=True)
            r.raise_for_status()
            with open(video_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded to {video_path}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to download from Discord: {e}", file=sys.stderr)
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
            sys.exit(1)

    job = Job(
        vibe=args.vibe,
        drive_url=args.drive_url,
        video_path=video_path,
        genre=args.genre,
        default_privacy=args.privacy,
        force_normal=args.force_normal
    )
    
    try:
        semaphore = asyncio.Semaphore(1)
        video_id = await process_job(job, semaphore)
        if video_id:
            print(f"SUCCESS: {video_id}")
        else:
            print("FAILED: Job completed but returned no video ID.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Job Failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if args.discord_url and video_path and os.path.exists(video_path):
            os.remove(video_path)
            print(f"Cleaned up temp file {video_path}", file=sys.stderr)

async def handle_list(args):
    from uploader import get_youtube_client
    loop = asyncio.get_event_loop()
    
    def _blocking_list():
        youtube = get_youtube_client()
        channels_response = youtube.channels().list(
            mine=True,
            part="contentDetails"
        ).execute()
        
        if not channels_response.get("items"):
            return []
            
        uploads_playlist_id = channels_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        
        # Fetch up to 50 videos
        res = youtube.playlistItems().list(
            playlistId=uploads_playlist_id,
            part="snippet,status",
            maxResults=50
        ).execute()
        
        playlist_items = res.get("items", [])

        # Collect video IDs to batch-fetch statistics
        video_ids = []
        basic_map = {}
        for item in playlist_items:
            snippet = item.get("snippet", {})
            status = item.get("status", {})
            vid = snippet.get("resourceId", {}).get("videoId")
            if vid:
                video_ids.append(vid)
                basic_map[vid] = {
                    "id": vid,
                    "title": snippet.get("title"),
                    "privacy": status.get("privacyStatus", "unknown"),
                    "publishedAt": snippet.get("publishedAt"),
                    "views": 0,
                    "likes": 0,
                    "comments": 0
                }

        # Batch fetch statistics (max 50 IDs per call, which matches our limit)
        if video_ids:
            stats_res = youtube.videos().list(
                id=",".join(video_ids),
                part="statistics"
            ).execute()
            for stat_item in stats_res.get("items", []):
                vid = stat_item["id"]
                stats = stat_item.get("statistics", {})
                if vid in basic_map:
                    basic_map[vid]["views"] = int(stats.get("viewCount", 0))
                    basic_map[vid]["likes"] = int(stats.get("likeCount", 0))
                    basic_map[vid]["comments"] = int(stats.get("commentCount", 0))

        videos = [basic_map[vid] for vid in video_ids if vid in basic_map]
        return videos

    try:
        videos = await loop.run_in_executor(None, _blocking_list)
        print(json.dumps({"status": "success", "videos": videos}))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

async def handle_setprivacy(args):
    from uploader import get_youtube_client
    loop = asyncio.get_event_loop()
    
    def _blocking_set_privacy():
        youtube = get_youtube_client()
        body = {
            "id": args.video_id,
            "status": {
                "privacyStatus": args.privacy
            }
        }
        res = youtube.videos().update(
            part="status",
            body=body
        ).execute()
        return res.get("id")

    try:
        res_id = await loop.run_in_executor(None, _blocking_set_privacy)
        print(f"SUCCESS: {res_id}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

async def handle_delete(args):
    from uploader import get_youtube_client
    loop = asyncio.get_event_loop()
    
    def _blocking_delete():
        youtube = get_youtube_client()
        youtube.videos().delete(id=args.video_id).execute()
        return True

    try:
        await loop.run_in_executor(None, _blocking_delete)
        print("SUCCESS")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

async def main():
    parser = argparse.ArgumentParser(description="CLI for managing YouTube videos and uploads")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # upload subcommand
    upload_parser = subparsers.add_parser("upload", help="Upload a video")
    upload_parser.add_argument('--vibe', required=True, help="The vibe of the video")
    upload_parser.add_argument('--drive_url', required=False, help="Google Drive link")
    upload_parser.add_argument('--discord_url', required=False, help="Direct Discord attachment link")
    upload_parser.add_argument('--genre', default='comedy', help="Genre for YouTube category")
    upload_parser.add_argument('--privacy', default='public', help="Privacy status")
    upload_parser.add_argument('--force_normal', action='store_true', help="Force upload as normal video without padding")

    # list subcommand
    list_parser = subparsers.add_parser("list", help="List uploaded videos")
    list_parser.add_argument('--sort', default='date', choices=['date', 'views', 'likes', 'comments'], help="Sort order for video list")

    # setprivacy subcommand
    setprivacy_parser = subparsers.add_parser("setprivacy", help="Set privacy status of a video")
    setprivacy_parser.add_argument('--video_id', required=True, help="YouTube video ID")
    setprivacy_parser.add_argument('--privacy', required=True, choices=['public', 'private', 'unlisted'], help="Privacy status")

    # delete subcommand
    delete_parser = subparsers.add_parser("delete", help="Delete a video")
    delete_parser.add_argument('--video_id', required=True, help="YouTube video ID")

    args = parser.parse_args()

    if args.command == "upload":
        await handle_upload(args)
    elif args.command == "list":
        await handle_list(args)
    elif args.command == "setprivacy":
        await handle_setprivacy(args)
    elif args.command == "delete":
        await handle_delete(args)

if __name__ == '__main__':
    asyncio.run(main())
