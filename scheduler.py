import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from metadata import generate_metadata_async
from uploader import UploadConfig, upload_video, DEFAULT_PRIVACY, DEFAULT_NOTIFY
from drive import download_video

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """
    One upload job.

    Args:
        video_path:  Path to the local video file (optional if drive_url is provided).
        vibe:        The niche/energy e.g. 'funny meme', 'gaming fail', 'satisfying'.
        drive_url:   Google Drive shareable link to the video (optional).
        genre:       Broad genre used to pick a YouTube category (comedy, gaming, etc.).
        default_privacy:     'private', 'unlisted', or 'public'. Defaults to 'private' — change when ready.
    """
    vibe: str
    video_path: Optional[str] = None
    drive_url: Optional[str] = None
    genre: str = "default"
    default_privacy: str = DEFAULT_PRIVACY
    force_normal: bool = False
    acc_id: Optional[str] = None
    notify_subscribers: bool = DEFAULT_NOTIFY

    def __post_init__(self):
        from uploader import GENRE_CATEGORY_MAP
        key = self.genre.lower().split("&")[0].strip()
        if key not in GENRE_CATEGORY_MAP and key != "default":
            valid = [k for k in GENRE_CATEGORY_MAP.keys() if k != "default"]
            raise ValueError(f"Invalid genre '{self.genre}'. Must be one of: {', '.join(valid)}")


async def process_job(job: Job, semaphore: asyncio.Semaphore) -> Optional[str]:
    """
    Full pipeline for one video: Gemini watches video + generates metadata → upload.
    """
    async with semaphore:
        active_video_path = job.video_path
        cleanup_needed = False
        
        try:
            # If a Google Drive URL is provided, download it first
            if job.drive_url:
                active_video_path = f"videos/temp_{id(job)}.mp4"
                logger.info(f"[START] Downloading from Drive: {job.drive_url}")
                success = await download_video(job.drive_url, active_video_path)
                if not success:
                    raise RuntimeError(f"Google Drive download failed for URL: {job.drive_url}")
                cleanup_needed = True
            
            if not active_video_path:
                raise ValueError("Neither video_path nor drive_url was provided.")

            logger.info(f"[START] Processing {active_video_path}")

            # Preprocessing: Check length and aspect ratio
            from video_processor import get_video_info, pad_video_for_shorts
            info = get_video_info(active_video_path)
            
            if info["duration"] == 0:
                if not job.force_normal:
                    logger.error(f"[ERROR] Could not read video info for {active_video_path}.")
                    print(f"FFPROBE_FAILED: Could not read video info. Is ffmpeg installed? Use force_normal to bypass.", file=sys.stderr)
                    raise RuntimeError("FFPROBE_FAILED")
                else:
                    logger.warning(f"[WARNING] Could not read video info for {active_video_path}. "
                                   "Skipping preprocessing — video will upload as-is due to force_normal.")
                    print(f"[WARNING] Could not read video info. Uploading as-is due to force_normal flag.", file=sys.stderr)
            elif info["duration"] > 180:
                if not job.force_normal:
                    if sys.stdin.isatty():
                        # Standalone interactive terminal mode
                        print(f"\n[WARNING] Video '{active_video_path}' is {info['duration']:.1f}s long (limit is 180s).")
                        print("YouTube will treat this as a NORMAL long-form video, NOT a Short.")
                        loop = asyncio.get_event_loop()
                        ans = await loop.run_in_executor(None, input, "Do you want to upload it as a normal video? (y/n): ")
                        if ans.strip().lower() != 'y':
                            raise RuntimeError("Upload aborted by user: video too long for Shorts.")
                    else:
                        # Non-interactive mode (e.g. running via Discord bot exec)
                        print("VIDEO_TOO_LONG: Video exceeds 180s limit for Shorts.", file=sys.stderr)
                        raise RuntimeError("VIDEO_TOO_LONG")
                else:
                    print(f"[WARNING] Video '{active_video_path}' is {info['duration']:.1f}s long. Uploading as normal video due to force_normal flag.")
            elif info["duration"] > 0:
                # If it's a valid duration <= 60s, ensure it is vertical
                if info["is_horizontal"]:
                    print(f"[PREPROCESS] Padding horizontal video {active_video_path}...")
                    padded_path = await asyncio.get_event_loop().run_in_executor(None, pad_video_for_shorts, active_video_path)
                    if padded_path != active_video_path:
                        if cleanup_needed and os.path.exists(active_video_path):
                            os.remove(active_video_path)
                        active_video_path = padded_path
                        cleanup_needed = True

            # Gemini watches the video and generates metadata
            metadata = await generate_metadata_async(active_video_path, job.vibe)

            if not metadata:
                raise RuntimeError(f"Metadata generation failed for {active_video_path}")

            logger.info(f"[META] Title: {metadata['title']}")
            logger.info(f"[META] Tags ({len(metadata['tags'])}): {', '.join(metadata['tags'][:5])}...")
            logger.info(f"[META] Thumbnail idea: {metadata['thumbnail_recommendation']}")

            # Upload at original quality
            config = UploadConfig(
                video_path=active_video_path,
                title=metadata["title"],
                description=metadata["description"],
                tags=metadata["tags"],
                genre=job.genre,
                default_privacy=job.default_privacy,
                acc_id=job.acc_id,
                notify_subscribers=job.notify_subscribers,
            )

            video_id = await upload_video(config)
            if video_id:
                logger.info(f"[DONE] Uploaded → https://www.youtube.com/shorts/{video_id}")
            else:
                raise RuntimeError(f"Upload failed for {active_video_path}")
            return video_id

        finally:
            if cleanup_needed and active_video_path and os.path.exists(active_video_path):
                os.remove(active_video_path)
                logger.info(f"[CLEANUP] Deleted temporary file {active_video_path}")


async def run_batch(jobs: list[Job], max_concurrent: int = 1) -> list[str]:
    """
    Upload multiple videos concurrently.
    max_concurrent=1 keeps you safely within YouTube's and Gemini's free tier quotas.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [process_job(job, semaphore) for job in jobs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = [r for r in results if isinstance(r, str)]
    failed     = [r for r in results if not isinstance(r, str)]

    for f in failed:
        if isinstance(f, Exception):
            logger.error(f"Job failed with exception: {f}", exc_info=f)

    logger.info(f"Batch complete — Uploaded: {len(successful)}, Failed: {len(failed)}")
    return successful
