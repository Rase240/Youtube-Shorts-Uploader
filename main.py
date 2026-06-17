import asyncio
import logging
from scheduler import Job, run_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def main():
    # ─────────────────────────────────────────────
    # Add your Shorts here.
    #
    # drive_url   → Google Drive shareable link to the video (or video_path for local file)
    # vibe        → the energy/niche: 'funny meme', 'gaming fail', 'satisfying', etc.
    # genre       → broad category: comedy, gaming, music, education, entertainment, etc.
    # privacy     → 'public' by default, or you can specify 'private' or 'unlisted'
    # ─────────────────────────────────────────────
    jobs = [
        Job(
            drive_url="https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing",  # Put your drive link here
            vibe="your video vibe/niche",
            genre="comedy",
        ),
        # You can still use local files too:
        # Job(
        #     video_path="videos/meme2.mp4",
        #     vibe="relatable animal meme",
        #     genre="comedy",
        # ),
    ]

    video_ids = await run_batch(jobs, max_concurrent=2)

    print("\nUploaded Short IDs:")
    for vid_id in video_ids:
        print(f"   https://www.youtube.com/shorts/{vid_id}")


if __name__ == "__main__":
    asyncio.run(main())
