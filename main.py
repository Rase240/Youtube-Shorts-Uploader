import asyncio
import logging
from scheduler import Job, run_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def main():
    # ─────────────────────────────────────────────
    # Add your Shorts here.
    #
    # drive_url      → Google Drive shareable link to the video (or video_path for local file)
    # content_brief  → instructions on how to frame the content (a simple string paragraph)
    # genre          → broad category: comedy, gaming, music, education, entertainment, etc.
    # default_privacy→ 'public' by default, or you can specify 'private' or 'unlisted'
    # ─────────────────────────────────────────────
    jobs = [
        Job(
            drive_url="https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing",  # Put your drive link here
            content_brief="Funny cat clip. Focus on the unexpected friendship between the cat and duck.",
            genre="comedy",
        ),
        # Example with a detailed multi-line brief:
        # Job(
        #     video_path="videos/meme2.mp4",
        #     content_brief=(
        #         "Funny animal meme. The joke is that the dog has an irrational fear "
        #         "of a kitchen drawer and freezes whenever it opens. Titles should "
        #         "feel like a friend texting a friend after seeing something weird. "
        #         "Avoid generic clickbait and reaction memes."
        #     ),
        #     genre="comedy",
        # ),
    ]

    video_ids = await run_batch(jobs, max_concurrent=2)

    print("\nUploaded Short IDs:")
    for vid_id in video_ids:
        print(f"   https://www.youtube.com/shorts/{vid_id}")


if __name__ == "__main__":
    asyncio.run(main())
