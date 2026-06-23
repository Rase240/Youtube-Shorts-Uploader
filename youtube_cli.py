import argparse
import asyncio
import requests
import uuid
import os
import json
import sys
async def handle_get_auth_url(args):
    from auth import get_authorization_url
    print(get_authorization_url())

async def handle_auth_with_code(args):
    from auth import get_credentials_from_code
    from account_manager import add_account, remove_account
    
    import uuid
    import shutil
    _PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    tmp_token = os.path.join(_PROJECT_DIR, f"token_tmp_{uuid.uuid4().hex[:8]}.pickle")
    try:
        get_credentials_from_code(args.code, tmp_token)
        # Fetch channel name
        from googleapiclient.discovery import build
        from auth import get_credentials
        creds = get_credentials(tmp_token)
        youtube = build("youtube", "v3", credentials=creds)
        res = youtube.channels().list(mine=True, part="snippet").execute()
        items = res.get("items", [])
        if items:
            channel_name = items[0]["snippet"]["title"]
        else:
            channel_name = "Unknown Channel"
            
        acc_id, new_token_file = add_account(channel_name)
        try:
            shutil.move(tmp_token, os.path.join(_PROJECT_DIR, new_token_file))
        except Exception as move_err:
            remove_account(acc_id)
            raise Exception(f"Failed to move token file, account creation rolled back: {move_err}")
            
        print(f"SUCCESS: {channel_name} (ID: {acc_id})")
    except Exception as e:
        if os.path.exists(tmp_token):
            os.remove(tmp_token)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

async def handle_acc_list(args):
    from account_manager import load_accounts, migrate_legacy_token
    migrate_legacy_token()
    data = load_accounts()
    print(json.dumps(data))

async def handle_set_acc(args):
    from account_manager import set_current_account
    try:
        set_current_account(args.acc_id)
        print("SUCCESS")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

async def handle_whoami(args):
    from account_manager import get_account_name
    name = get_account_name(args.acc)
    print(json.dumps({"channel_name": name}))

async def handle_sync_acc(args):
    from googleapiclient.discovery import build
    from auth import get_credentials
    from account_manager import load_accounts, update_account_name
    try:
        data = load_accounts()
        target_id = str(args.acc_id) if getattr(args, 'acc_id', None) else str(data.get("current_account"))
        if target_id not in data.get("accounts", {}):
            print("ERROR: Account not found.", file=sys.stderr)
            sys.exit(1)
            
        token_file = data["accounts"][target_id]["token_file"]
        _PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
        token_path = os.path.join(_PROJECT_DIR, token_file)
        
        creds = get_credentials(token_path)
        youtube = build("youtube", "v3", credentials=creds)
        res = youtube.channels().list(mine=True, part="snippet").execute()
        items = res.get("items", [])
        if items:
            channel_name = items[0]["snippet"]["title"]
            update_account_name(target_id, channel_name)
            print(f"SUCCESS: {channel_name} (ID: {target_id})")
        else:
            print("ERROR: Could not fetch channel name.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

async def handle_upload(args):
    video_path = None
    if args.discord_url:
        os.makedirs("videos", exist_ok=True)
        video_path = f"videos/temp_discord_{uuid.uuid4().hex[:8]}.mp4"
        print(f"Downloading from Discord: {args.discord_url}", file=sys.stderr)
        try:
            r = requests.get(args.discord_url, stream=True, timeout=30)
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

    from scheduler import Job, process_job
    notify_val = None
    if getattr(args, 'notify', None) is not None:
        notify_val = args.notify.lower() in ('true', '1', 'yes')

    job_kwargs = {
        "vibe": args.vibe,
        "drive_url": args.drive_url,
        "video_path": video_path,
        "genre": args.genre,
        "default_privacy": args.privacy,
        "force_normal": args.force_normal,
        "acc_id": args.acc
    }
    if notify_val is not None:
        job_kwargs["notify_subscribers"] = notify_val

    job = Job(**job_kwargs)
    
    try:
        semaphore = asyncio.Semaphore(1)
        video_id = await process_job(job, semaphore)
        from account_manager import get_account_name
        channel_name = get_account_name(args.acc)
        if video_id:
            print(f"SUCCESS: [{channel_name}] {video_id}")
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
        youtube = get_youtube_client(args.acc)
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
        
        sort_by = getattr(args, "sort", "date")
        if sort_by == "views":
            videos.sort(key=lambda x: x["views"], reverse=True)
        elif sort_by == "likes":
            videos.sort(key=lambda x: x["likes"], reverse=True)
        elif sort_by == "comments":
            videos.sort(key=lambda x: x["comments"], reverse=True)
        else:
            videos.sort(key=lambda x: x["publishedAt"] or "", reverse=True)
            
        return videos

    try:
        from account_manager import get_account_name
        channel_name = get_account_name(args.acc)
        videos = await loop.run_in_executor(None, _blocking_list)
        print(json.dumps({"status": "success", "channel_name": channel_name, "videos": videos}))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

async def handle_setprivacy(args):
    from uploader import get_youtube_client
    loop = asyncio.get_event_loop()
    
    def _blocking_set_privacy():
        youtube = get_youtube_client(args.acc)
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
        from account_manager import get_account_name
        channel_name = get_account_name(args.acc)
        print(f"SUCCESS: [{channel_name}] {res_id}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

async def handle_delete(args):
    from uploader import get_youtube_client
    loop = asyncio.get_event_loop()
    
    def _blocking_delete():
        youtube = get_youtube_client(args.acc)
        youtube.videos().delete(id=args.video_id).execute()
        return True

    try:
        await loop.run_in_executor(None, _blocking_delete)
        from account_manager import get_account_name
        channel_name = get_account_name(args.acc)
        print(f"SUCCESS: [{channel_name}]")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

async def handle_mt(args):
    import os
    import requests
    import uuid
    import json
    import sys
    from metadata import generate_metadata_async
    
    video_path = getattr(args, 'video_path', None)
    cleanup_video = False

    if getattr(args, 'discord_url', None):
        os.makedirs("videos", exist_ok=True)
        video_path = f"videos/temp_discord_{uuid.uuid4().hex[:8]}.mp4"
        print(f"Downloading from Discord: {args.discord_url}", file=sys.stderr)
        try:
            r = requests.get(args.discord_url, stream=True, timeout=30)
            r.raise_for_status()
            with open(video_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded to {video_path}", file=sys.stderr)
            cleanup_video = True
        except Exception as e:
            print(f"Failed to download from Discord: {e}", file=sys.stderr)
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
            sys.exit(1)

    if not video_path:
        print("ERROR: Must provide either --video_path or --discord_url", file=sys.stderr)
        sys.exit(1)

    try:
        metadata = await generate_metadata_async(video_path, args.vibe)
        if metadata:
            print(json.dumps(metadata, indent=2))
        else:
            print("ERROR: Metadata generation returned None", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if cleanup_video and video_path and os.path.exists(video_path):
            os.remove(video_path)
            print(f"Cleaned up temp file {video_path}", file=sys.stderr)

async def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--acc', required=False, help="Account ID to use")

    parser = argparse.ArgumentParser(description="CLI for managing YouTube videos and uploads")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # upload subcommand
    upload_parser = subparsers.add_parser("upload", parents=[common], help="Upload a video")
    upload_parser.add_argument('--vibe', required=True, help="The vibe of the video")
    upload_parser.add_argument('--drive_url', required=False, help="Google Drive link")
    upload_parser.add_argument('--discord_url', required=False, help="Direct Discord attachment link")
    upload_parser.add_argument('--genre', default='comedy', help="Genre for YouTube category")
    upload_parser.add_argument('--privacy', default='public', choices=['public', 'private', 'unlisted'], help="Privacy status")
    upload_parser.add_argument('--force_normal', action='store_true', help="Force upload as normal video without padding")
    upload_parser.add_argument('--notify', default=None, choices=['true', 'false'], help="Notify subscribers")

    # list subcommand
    list_parser = subparsers.add_parser("list", parents=[common], help="List uploaded videos")
    list_parser.add_argument('--sort', default='date', choices=['date', 'views', 'likes', 'comments'], help="Sort order for video list")

    # mt subcommand
    mt_parser = subparsers.add_parser("mt", aliases=['metadata'], parents=[common], help="Generate metadata for a video without uploading")
    mt_parser.add_argument('--vibe', required=True, help="The vibe of the video")
    mt_parser.add_argument('--video_path', required=False, help="Local path to the video file")
    mt_parser.add_argument('--discord_url', required=False, help="Direct Discord attachment link")

    # setprivacy subcommand
    setprivacy_parser = subparsers.add_parser("setprivacy", parents=[common], help="Set privacy status of a video")
    setprivacy_parser.add_argument('--video_id', required=True, help="YouTube video ID")
    setprivacy_parser.add_argument('--privacy', required=True, choices=['public', 'private', 'unlisted'], help="Privacy status")

    # delete subcommand
    delete_parser = subparsers.add_parser("delete", parents=[common], help="Delete a video")
    delete_parser.add_argument('--video_id', required=True, help="YouTube video ID")

    # multiple accounts subcommands
    subparsers.add_parser("get_auth_url", help="Get OAuth authorization URL")
    auth_with_code_parser = subparsers.add_parser("auth_with_code", help="Complete OAuth with code")
    auth_with_code_parser.add_argument('--code', required=True, help="Authorization code")
    subparsers.add_parser("acc_list", help="List all accounts")
    set_acc_parser = subparsers.add_parser("set_acc", help="Set current account")
    set_acc_parser.add_argument('--acc_id', required=True, help="Account ID to set as current")
    subparsers.add_parser("whoami", parents=[common], help="Get current account name")
    
    sync_acc_parser = subparsers.add_parser("sync_acc", help="Sync/Update channel name from YouTube API")
    sync_acc_parser.add_argument('--acc_id', required=False, help="Account ID to sync (defaults to current)")

    args = parser.parse_args()

    if args.command == "upload":
        if args.drive_url and args.discord_url:
            parser.error("Cannot specify both --drive_url and --discord_url")
        await handle_upload(args)
    elif args.command == "list":
        await handle_list(args)
    elif args.command in ["mt", "metadata"]:
        await handle_mt(args)
    elif args.command == "setprivacy":
        await handle_setprivacy(args)
    elif args.command == "delete":
        await handle_delete(args)
    elif args.command == "get_auth_url":
        await handle_get_auth_url(args)
    elif args.command == "auth_with_code":
        await handle_auth_with_code(args)
    elif args.command == "acc_list":
        await handle_acc_list(args)
    elif args.command == "set_acc":
        await handle_set_acc(args)
    elif args.command == "whoami":
        await handle_whoami(args)
    elif args.command == "sync_acc":
        await handle_sync_acc(args)

if __name__ == '__main__':
    asyncio.run(main())
