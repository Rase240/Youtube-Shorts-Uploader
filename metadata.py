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
    video_analysis: str = Field(
        ...,
        description=(
            "Briefly analyze the video's visual hooks, pacing, and core message. "
            "Identify the most striking moment or conflict that will capture attention. "
            "This is your 'scratchpad' to think before generating the metadata."
        )
    )
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
            "A hyper-engaging, attention-grabbing YouTube Shorts title under 55 characters. "
            "Use extreme curiosity gaps, bold claims, relatable pain points, or negative hooks. "
            "Front-load the most important keywords. Focus on the core conflict or punchline. "
            "Avoid boring descriptive titles. Lowercase and emojis are allowed if they fit the vibe."
        )
    )
    description: str = Field(
        ...,
        description=(
            "A concise YouTube Shorts description. "
            "Line 1 MUST be a hard-hitting hook or controversial question. "
            "Lines 2-3 should provide brief, natural-sounding context that weaves in SEO keywords. "
            "End with exactly 5-7 highly specific, algorithm-friendly hashtags on a single line. "
            "CRITICAL: NEVER use generic, spammy, or cringy hashtags like #viral, #funnymemes, #trending, #fyp, or #Shorts. "
            "Only use strictly relevant, high-search-volume niche tags (e.g., #CarRestoration, #BakingFails). "
            "Never use generic AI intros like 'In this video...' or 'Welcome back...'."
        )
    )
    tags: list[str] = Field(
        ...,
        description=(
            "10-15 high-search-volume YouTube tags relevant to the video (no # symbols). "
            "CRITICAL: Ban all generic tags (viral, funny, meme, trending, shorts). "
            "Focus ONLY on strict SEO entity keywords. If the video is about a dog eating pizza, "
            "tags should be 'golden retriever', 'dog eating human food', 'funny dog compilation', NOT 'funnymemes'."
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
    pinned_comment_suggestion: str = Field(
        ...,
        description=(
            "A highly engaging question or controversial statement to pin in the comments section. "
            "This must force viewers to reply. Examples: 'Would you have reacted the same way?' or "
            "'Who do you think was in the wrong here?' Keep it short and conversational."
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
You are a top-tier Gen-Z YouTube Shorts strategist and growth hacker, famous for viral hooks and massive CTRs.
I have attached a video for you to watch.

The intended vibe/niche is: {vibe}

Follow these steps:
1. ANALYZE: Watch the video carefully. Use the `video_analysis` field to break down the most satisfying, funny, or controversial moment.
2. HOOK: Determine the `target_emotion` and `hook_style`.
3. TITLES & METADATA: Craft metadata that absolutely forces a viewer to stop scrolling. 

Strict Constraints for Titles & Descriptions:
- TITLES MUST BE EXTREMELY CLICKABLE. Use strong curiosity gaps (e.g., "I tried...", "Why you shouldn't..."), extreme outcomes, or highly relatable scenarios. NO boring literal descriptions.
- NEVER use generic AI buzzwords: 'unleash', 'dive in', 'delve', 'testament', 'ultimate guide', 'revolutionize', 'look no further', 'mastering', 'epic'.
- HASHTAGS MUST BE NICHE-SPECIFIC. Do not use generic tags like #viral, #funnymemes, #trending, #shorts. Use exact entities (e.g., #Woodworking, #GoldenRetriever).
- PINNED COMMENT: Must be a provocative, relatable, or highly opinionated question that forces viewers to type a reply.
- Title length: Keep it under 55 characters to avoid mobile truncation.
- Tone: Write exactly how a real native Shorts creator would talk. NO formal greetings or meta-commentary (e.g., "Check out this video!").
- Formatting: Use natural pacing. Emojis are okay but don't overdo them. Lowercase is great if it fits the Gen-Z aesthetic.

Example of BAD Title: "A Funny Dog Playing With A Ball #Shorts"
Example of GOOD Title: "he actually thought this would work 💀" or "the ending was personal..."
"""
            response = None
            parsed_data = None
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
                        if response and response.text:
                            parsed_data = json.loads(response.text)
                            success = True
                            break
                        else:
                            logger.warning(f"[GEMINI] Empty response text on {model_name} attempt {attempt + 1}/3.")
                    except APIError as e:
                        if "UNAVAILABLE" in str(e).upper():
                            if attempt < 2:
                                logger.warning(f"[GEMINI] {model_name} unavailable, retrying in 5s ({attempt + 1}/3)...")
                                await asyncio.sleep(5)
                            else:
                                logger.error(f"[GEMINI] {model_name} failed after 3 attempts.")
                        else:
                            raise e
                    except json.JSONDecodeError as je:
                        logger.warning(f"[GEMINI] Failed to parse JSON on attempt {attempt + 1}/3: {je}")
                        try:
                            logger.warning(f"Raw response text that failed:\n{response.text}")
                        except Exception:
                            pass
                        if attempt >= 2:
                            raise je
                
                if success:
                    break

            if parsed_data:
                return parsed_data
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
        try:
            if 'response' in locals() and response and hasattr(response, 'text'):
                logger.error(f"Raw Gemini response that failed to parse:\n{response.text}")
        except Exception:
            pass
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
