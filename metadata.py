import asyncio
import json
import logging
import os
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field, field_validator

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class YouTubeShortMetadata(BaseModel):
    title: str = Field(
        ...,
        description=(
            "A punchy, viral YouTube Shorts title under 100 characters. "
            "Make it curiosity-driven, emotional, or shocking. No clickbait fluff — "
            "it should reflect the actual vibe of the video."
        )
    )
    description: str = Field(
        ...,
        description=(
            "A 150-250 word YouTube Shorts description based on watching the video. "
            "Open with a strong hook sentence. "
            "Naturally weave in searchable keywords for the niche. "
            "End with 10-15 trending hashtags on separate lines (e.g. #Memes #Funny). "
            "Keep the tone matching the video — funny, hype, relatable, etc."
        )
    )
    tags: list[str] = Field(
        ...,
        description=(
            "12-20 high-search-volume YouTube tags relevant to the video. "
            "Mix broad tags with niche-specific ones based on the video content. "
            "Do NOT include the # symbol in tags."
        )
    )
    thumbnail_recommendation: str = Field(
        ...,
        description=(
            "A recommendation for the perfect custom thumbnail. "
            "Suggest the exact timestamp to pull the frame from (e.g. '00:04 where the cat jumps') "
            "and describe any text or graphics that should be overlaid to maximize CTR."
        )
    )

    @field_validator("title")
    @classmethod
    def cap_title(cls, v: str) -> str:
        return v[:97] + "..." if len(v) > 100 else v

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
                    return None
                await asyncio.sleep(2)

            logger.info("[GEMINI] Video ready. Generating metadata...")

            # 3. Generate content
            prompt = f"""
You are a viral YouTube Shorts growth expert. I have attached a video for you to watch.
The intended vibe/niche is: {vibe}

Please watch the video carefully and generate metadata that maximises CTR, watch time, and engagement.

Rules:
- Title must feel native to the platform — punchy, short, conversational.
- Description must include 10-15 relevant hashtags at the end (one per line).
- Tags should cover what people actually search for in this niche.
- Suggest a specific timestamp and overlay text for a custom thumbnail.
- Do NOT mention "clipping channel", "faceless channel", or anything about automation.
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
                return None

        finally:
            # 4. Clean up the file to save quota space
            logger.info("[GEMINI] Deleting video from Google's servers...")
            await client.aio.files.delete(name=video_file.name)

    except APIError as e:
        logger.error(f"Gemini API error: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini output as JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating metadata: {e}")
        return None


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
