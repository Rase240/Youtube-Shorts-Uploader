import asyncio
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field, field_validator

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_PROJECT_DIR, ".env"))

# Set up logging to both console and a rotating file
_LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
_LOG_FILE = os.path.join(_PROJECT_DIR, "youtube_bot.log")

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Prevent duplicate handlers
if not root_logger.handlers:
    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root_logger.addHandler(console_handler)

    # Rotating File handler (max 5MB, keeping 3 backups)
    try:
        file_handler = RotatingFileHandler(_LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root_logger.addHandler(file_handler)
    except Exception as e:
        sys.stderr.write(f"Failed to initialize file logging: {e}\n")

logger = logging.getLogger(__name__)


class YouTubeShortMetadata(BaseModel):
    target_emotion: str = Field(
        ...,
        description="The primary emotion the video triggers (e.g. curiosity, amusement, outrage, nostalgia, awe)."
    )
    hook_style: str = Field(
        ...,
        description="The hook technique used (e.g. 'unanswered question', 'pattern interrupt', 'relatable scenario')."
    )
    title: str = Field(
        ...,
        description=(
            "A punchy, viral YouTube Shorts title under 55 characters. "
            "Start with the most emotional/curiosity-driven words. No clickbait fluff — "
            "it should reflect the actual vibe of the video."
        )
    )
    description: str = Field(
        ...,
        description=(
            "A concise 100-150 word YouTube Shorts description. "
            "Line 1 MUST be a high-impact hook. Lines 2-4 should weave in searchable niche keywords naturally. "
            "End with exactly 4-7 highly relevant hashtags on separate lines (e.g. #Shorts, #Topic). "
            "Do not use generic intros like 'In this video...' or 'Welcome back...'"
        )
    )
    tags: list[str] = Field(
        ...,
        description=(
            "10-15 high-search-volume YouTube tags relevant to the video (no # symbols). "
            "Mix broad tags with niche-specific ones based on the video content."
        )
    )
    thumbnail_recommendation: str = Field(
        ...,
        description=(
            "A recommendation for the perfect custom thumbnail. "
            "Suggest the exact timestamp to pull the frame from (e.g. '00:04 where the cat jumps') "
            "and describe any text/graphics that should be overlaid to maximize CTR."
        )
    )

    @field_validator("title")
    @classmethod
    def cap_title(cls, v: str) -> str:
        return v[:52] + "..." if len(v) > 55 else v

    @field_validator("tags")
    @classmethod
    def cap_tags(cls, v: list[str]) -> list[str]:
        # YouTube allows up to 500 chars total for tags
        return [t.strip("#").strip() for t in v]


def get_gemini_client() -> genai.Client:
    return genai.Client()


async def generate_metadata_async(video_path: str, vibe: str) -> Optional[dict]:
    """
    Generate engagement-optimised YouTube metadata by having Gemini actually watch the video.

    Args:
        video_path: Path to the video file to upload.
        vibe:       The vibe/niche e.g. 'funny meme', 'gaming fail', 'relatable moment'.

    Returns:
        Dict with keys: title, description, tags, thumbnail_recommendation — or None on failure.
    """
    client = get_gemini_client()

    logger.info(f"[GEMINI] Uploading video {video_path} for processing...")
    try:
        # 1. Upload the video
        video_file = await client.aio.files.upload(file=video_path)
        logger.info(f"[GEMINI] Uploaded. Waiting for processing to finish (this takes a few seconds)...")

        try:
            # 2. Poll until ACTIVE
            while True:
                video_file = await client.aio.files.get(name=video_file.name)
                if video_file.state == "ACTIVE":
                    break
                elif video_file.state == "FAILED":
                    logger.error("[GEMINI] Video processing failed on Google's end.")
                    raise RuntimeError("Video processing failed on Google's end.")
                await asyncio.sleep(2)

            logger.info("[GEMINI] Video ready. Generating metadata...")

            # 3. Generate content
            prompt = f"""
You are an expert YouTube Shorts content strategist specializing in viral organic growth.
I have attached a video for you to watch.

The intended vibe/niche is: {vibe}

Please watch the video carefully and generate metadata that maximises CTR, watch time, and engagement.

Strict Constraints:
- NEVER use generic AI buzzwords: 'unleash', 'dive in', 'delve', 'testament', 'ultimate guide', 'revolutionize', 'look no further', 'mastering', 'nestled'.
- Keep titles under 55 characters so they do not get truncated on mobile screens.
- Do NOT use formal greetings or meta-commentary (e.g. 'Check out this video!'). Write exactly how a real creator or viewer would talk.
- Do NOT mention clipping, automation, or bot channels.
"""
            response = None
            models_to_try = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]
            
            for model_name in models_to_try:
                success = False
                for attempt in range(3):
                    try:
                        logger.info(f"[GEMINI] Generating metadata with {model_name} (Attempt {attempt + 1}/3)...")
                        response = await client.aio.models.generate_content(
                            model=model_name,
                            contents=[video_file, prompt],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=YouTubeShortMetadata,
                                max_output_tokens=1500,
                            ),
                        )
                        success = True
                        break
                    except APIError as e:
                        if "UNAVAILABLE" in str(e).upper():
                            if attempt < 2:
                                logger.warning(f"[GEMINI] {model_name} unavailable, retrying in 5s ({attempt + 1}/3)...")
                                await asyncio.sleep(5)
                            else:
                                logger.error(f"[GEMINI] {model_name} failed after 3 attempts.")
                        else:
                            raise e
                
                if success:
                    break

            if response and response.text:
                return json.loads(response.text)
            else:
                logger.error("Empty or failed response from Gemini after trying all models.")
                raise RuntimeError("Empty or failed response from Gemini after trying all models.")

        finally:
            # 4. Clean up the file to save quota space
            logger.info("[GEMINI] Deleting video from Google's servers...")
            await client.aio.files.delete(name=video_file.name)

    except APIError as e:
        logger.error(f"Gemini API error: {e}")
        raise RuntimeError(f"Gemini API error: {e}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini output as JSON: {e}")
        raise RuntimeError(f"Failed to parse Gemini output as JSON: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error generating metadata: {e}")
        raise RuntimeError(f"Unexpected error generating metadata: {e}") from e


# --- Quick test ---
async def _test():
    result = await generate_metadata_async(
        video_path="videos/meme1.mp4",
        vibe="funny animal meme",
    )
    if result:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(_test())
