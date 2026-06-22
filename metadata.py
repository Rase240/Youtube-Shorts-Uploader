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
from pydantic import BaseModel, Field, ValidationError, field_validator

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



class VideoAnalysis(BaseModel):
    """Phase 1 output: Deep analysis of the video content."""
    key_moment: str = Field(
        ...,
        description=(
            "Describe the single most striking, funny, shocking, or satisfying moment in the video. "
            "Be specific — reference what happens visually and when it occurs."
        )
    )
    emotional_arc: str = Field(
        ...,
        description=(
            "Describe the emotional journey a viewer goes through watching this video. "
            "What do they feel at the start vs. the climax vs. the end?"
        )
    )
    shareability_factor: str = Field(
        ...,
        description=(
            "Why would someone send this to a friend? What makes it worth sharing? "
            "Is it relatable, shocking, funny, wholesome, or rage-inducing?"
        )
    )
    core_hook: str = Field(
        ...,
        description=(
            "In one sentence, what is the irresistible hook of this video? "
            "What makes it impossible to scroll past?"
        )
    )
    subject_entities: list[str] = Field(
        ...,
        description=(
            "List 3-5 specific entities/subjects visible in the video "
            "(e.g. 'golden retriever', 'skateboard', 'kitchen', 'street food vendor'). "
            "These will be used for SEO tags later."
        )
    )


class TitleCandidates(BaseModel):
    """Phase 2 output: Multiple title options ranked by quality."""
    candidates: list[str] = Field(
        ...,
        description=(
            "Generate exactly 5 unique title candidates for this YouTube Short. "
            "Each must be under 55 characters, lowercase Gen-Z voice, and use different hook techniques "
            "(curiosity gap, relatable scenario, pattern interrupt, bold claim, emotional reaction). "
            "NO emojis unless they genuinely add to the vibe — most titles work better without them."
        )
    )
    ranking_reasoning: str = Field(
        ...,
        description=(
            "Briefly explain why you ranked them in this order. "
            "What makes #1 the strongest? Why are the others weaker?"
        )
    )
    best_title: str = Field(
        ...,
        description=(
            "The single best title from your candidates list. Copy it exactly. "
            "This MUST be under 55 characters."
        )
    )

    @field_validator("best_title")
    @classmethod
    def cap_title(cls, v: str) -> str:
        if len(v) > 55:
            truncated = v[:52]
            # Cut at last space to avoid chopping mid-word
            last_space = truncated.rfind(" ")
            if last_space > 20:
                truncated = truncated[:last_space]
            return truncated + "..."
        return v


class SupportingMetadata(BaseModel):
    """Phase 3 output: Description, tags, and engagement metadata."""
    description: str = Field(
        ...,
        description=(
            "A concise YouTube Shorts description. "
            "Line 1 MUST be a hard-hitting hook or controversial question. "
            "Lines 2-3 should provide brief, natural-sounding context that weaves in SEO keywords. "
            "End with 10 to 13 hashtags total on a single line, ordered niche-first then mainstream. "
            "Vary the exact total and the split slightly from video to video instead of landing on the "
            "same numbers every time. Roughly 55-65% should be NICHE hashtags — specific entity/character/"
            "subject tags tied directly to this video "
            "(e.g. #SpecificTopic, #NicheSubject, #TopicVariation). "
            "The remainder should be MAINSTREAM hashtags — broad discovery tags real high-performing videos "
            "in this category use for top-of-funnel reach (e.g. #broadcategory, #shorts, #relatable, "
            "#trending — pick whichever genuinely fit the video's category/vibe). "
            "Never use generic AI intros like 'In this video...' or 'Welcome back...'."
        )
    )
    niche_hashtag_count: int = Field(
        ...,
        description=(
            "The exact number of hashtags, counting from the START of the hashtag line in `description`, "
            "that are NICHE hashtags (before the mainstream ones begin). This MUST match how you actually "
            "ordered the hashtag line — e.g. if the first 8 hashtags are niche and the last 5 are mainstream, "
            "this value is 8. Used to verify the niche-first ordering and the 55-65% split programmatically."
        )
    )
    tags: list[str] = Field(
        ...,
        description=(
            "12 to 15 YouTube tags (no # symbols), ordered niche-first then mainstream. "
            "Vary the exact total and the split slightly from video to video instead of landing on the "
            "same numbers every time. Roughly 55-65% should be NICHE tags — specific multi-word search "
            "phrases tied to the exact subjects/entities/characters in this video "
            "(e.g. 'specific topic description', 'niche action phrase', 'subject variation'). "
            "The remainder should be MAINSTREAM tags — broader single-word or short-phrase category tags "
            "that real high-view videos in this niche rank for, used purely for discovery reach "
            "(e.g. 'broad category', 'category shorts', 'viral humor', 'relatable concept' "
            "— pick whichever fit this video's category). "
            "Every tag, niche or mainstream, must be something a real person would plausibly type into YouTube search."
        )
    )
    niche_tag_count: int = Field(
        ...,
        description=(
            "The exact number of tags, counting from the START of the `tags` list, that are NICHE tags "
            "(before the mainstream ones begin). This MUST match how you actually ordered the `tags` list — "
            "e.g. if tags[0:9] are niche and tags[9:13] are mainstream, this value is 9. "
            "Used to verify the niche-first ordering and the 55-65% split programmatically."
        )
    )
    thumbnail_recommendation: str = Field(
        ...,
        description=(
            "Suggest the exact timestamp to pull the thumbnail frame from "
            "(e.g. '00:04 where the cat jumps') and describe any text/graphics overlay to maximize CTR."
        )
    )
    pinned_comment_suggestion: str = Field(
        ...,
        description=(
            "A highly engaging question or controversial statement to pin in the comments. "
            "Must force viewers to reply. Keep it under 15 words and conversational."
        )
    )

    @field_validator("tags")
    @classmethod
    def cap_tags(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip("#").strip() for t in v]
        # Hard upper cap of 15 tags — matches the top of the 12-15 target range.
        if len(cleaned) > 15:
            logger.warning(
                f"[PHASE 3] Model returned {len(cleaned)} tags, truncating to 15. "
                f"Dropped: {cleaned[15:]}"
            )
            return cleaned[:15]
        return cleaned


# --- Title Quality Gate ---
_BANNED_TITLE_WORDS = {
    "unleash", "epic", "ultimate", "revolutionary", "incredible", "amazing",
    "unbelievable", "mind-blowing", "jaw-dropping", "insane", "you won't believe",
    "wait for it", "watch until the end", "must see", "gone wrong", "gone viral",
    "#shorts", "subscribe", "like and subscribe", "delve", "dive in",
    "look no further", "mastering", "testament", "revolutionize",
}


def _check_title_quality(title: str) -> Optional[str]:
    """Returns a rejection reason if the title is low quality, None if it passes."""
    if not title or len(title.strip()) < 10:
        return f"Title too short ({len(title.strip())} chars)"

    if len(title.strip()) > 55:
        return f"Title too long ({len(title.strip())} chars)"

    title_lower = title.lower()
    for banned in _BANNED_TITLE_WORDS:
        if banned in title_lower:
            return f"Contains banned word/phrase: '{banned}'"

    # Reject overly formal/capitalized titles (e.g. "A Funny Dog Playing With A Ball")
    words = title.split()
    if len(words) >= 4:
        capitalized = sum(1 for w in words if w[0].isupper() and len(w) > 1)
        if capitalized / len(words) > 0.7:
            return "Title looks too formal/capitalized (not Gen-Z voice)"

    return None


def _log_hashtag_and_tag_counts(
    description: str,
    tags: list[str],
    niche_hashtag_count: int,
    niche_tag_count: int,
) -> None:
    """Logs hashtag/tag counts AND verifies the claimed niche/mainstream split against the
    actual ordering, so the 55-65% niche ratio is checked in code instead of pure prompt-trust.
    Non-fatal — just visibility, since Gemini won't always land inside the target range."""
    import re
    hashtags = re.findall(r"#\w+", description)
    n_hashtags = len(hashtags)
    n_tags = len(tags)

    logger.info(
        f"[PHASE 3] Hashtag count: {n_hashtags} (target range 10-13) | "
        f"Tag count: {n_tags} (target range 12-15, hard cap 15)"
    )
    if n_hashtags < 10 or n_hashtags > 13:
        logger.warning(f"[PHASE 3] Hashtag count outside target range: got {n_hashtags} — {hashtags}")
    if n_tags < 12:
        logger.warning(f"[PHASE 3] Tag count below target range: got {n_tags} — {tags}")

    # Verify the niche:mainstream ratio against the model's self-reported split point,
    # rather than just trusting the prompt instruction blindly.
    if n_hashtags > 0:
        # Clamp in case the model's self-reported count exceeds the actual hashtag count.
        effective_niche_hashtag_count = min(niche_hashtag_count, n_hashtags)
        hashtag_ratio = effective_niche_hashtag_count / n_hashtags
        logger.info(
            f"[PHASE 3] Hashtag niche split: {effective_niche_hashtag_count}/{n_hashtags} "
            f"({hashtag_ratio:.0%} niche, target 55-65%)"
        )
        if not (0.50 <= hashtag_ratio <= 0.70):
            logger.warning(
                f"[PHASE 3] Hashtag niche ratio outside expected range: "
                f"{effective_niche_hashtag_count}/{n_hashtags} = {hashtag_ratio:.0%}"
            )
        if niche_hashtag_count > n_hashtags:
            logger.warning(
                f"[PHASE 3] niche_hashtag_count ({niche_hashtag_count}) exceeds actual hashtag "
                f"count ({n_hashtags}) — likely model inconsistency."
            )

    if n_tags > 0:
        # Clamp in case the model's self-reported split point predates the cap_tags truncation.
        effective_niche_tag_count = min(niche_tag_count, n_tags)
        tag_ratio = effective_niche_tag_count / n_tags
        logger.info(
            f"[PHASE 3] Tag niche split: {effective_niche_tag_count}/{n_tags} "
            f"({tag_ratio:.0%} niche, target 55-65%)"
        )
        if not (0.50 <= tag_ratio <= 0.70):
            logger.warning(
                f"[PHASE 3] Tag niche ratio outside expected range: "
                f"{effective_niche_tag_count}/{n_tags} = {tag_ratio:.0%}"
            )
        if niche_tag_count > n_tags:
            logger.warning(
                f"[PHASE 3] niche_tag_count ({niche_tag_count}) exceeds actual tag "
                f"count ({n_tags}) — likely due to cap_tags truncation or model inconsistency."
            )


def get_gemini_client() -> genai.Client:
    return genai.Client()


async def _call_gemini(client, model: str, contents, schema, max_tokens: int, temperature: float, max_attempts: int = 3):
    """Helper to call Gemini with retries and model fallback."""
    for attempt in range(max_attempts):
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            if response and response.text:
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                parsed = json.loads(text)
                validated = schema.model_validate(parsed)
                return validated.model_dump()
            else:
                logger.warning(f"[GEMINI] Empty response from {model} on attempt {attempt + 1}/{max_attempts}.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2)
        except APIError as e:
            error_str = str(e).upper()
            if ("UNAVAILABLE" in error_str or "429" in error_str or "RESOURCE_EXHAUSTED" in error_str) and attempt < max_attempts - 1:
                wait_time = 15 if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str else 5
                logger.warning(f"[GEMINI] {model} rate-limited/unavailable, retrying in {wait_time}s ({attempt + 1}/{max_attempts})...")
                await asyncio.sleep(wait_time)
            else:
                raise
        except (json.JSONDecodeError, ValidationError) as je:
            logger.warning(f"[GEMINI] Validation or JSON parse failed on attempt {attempt + 1}/{max_attempts}: {je}")
            if attempt >= max_attempts - 1:
                raise
            await asyncio.sleep(2)
        except Exception as e:
            logger.warning(f"[GEMINI] Unexpected error on attempt {attempt + 1}/{max_attempts}: {e}")
            if attempt >= max_attempts - 1:
                raise
            await asyncio.sleep(2)
    return None


async def generate_metadata_async(video_path: str, vibe: str) -> Optional[dict]:
    """
    3-phase metadata generation pipeline:
      Phase 1 (Analysis):  gemini-3.5-flash watches the video → deep analysis
      Phase 2 (Title):     gemini-3.5-flash crafts 5 title candidates → picks best
      Phase 3 (Metadata):  gemini-3.5-flash generates description, tags, etc.

    Args:
        video_path: Path to the video file to upload.
        vibe:       The vibe/niche e.g. 'funny meme', 'gaming fail', 'relatable moment'.

    Returns:
        Dict with keys: title, description, tags, thumbnail_recommendation, pinned_comment_suggestion,
        plus analysis fields — or None on failure.
    """
    client = get_gemini_client()
    video_file = None

    logger.info(f"[GEMINI] Uploading video {video_path} for processing...")
    try:
        # Upload the video
        video_file = await client.aio.files.upload(file=video_path)
        logger.info(f"[GEMINI] Uploaded. Waiting for processing to finish...")

        try:
            # Poll until ACTIVE (timeout after 120s to prevent infinite hang)
            max_polls = 60  # 60 * 2s = 120 seconds max
            for poll_attempt in range(max_polls):
                video_file_info = await client.aio.files.get(name=video_file.name)
                if video_file_info.state == "ACTIVE":
                    break
                elif video_file_info.state == "FAILED":
                    logger.error("[GEMINI] Video processing failed on Google's end.")
                    raise RuntimeError("Video processing failed on Google's end.")
                await asyncio.sleep(2)
            else:
                logger.error(f"[GEMINI] Video processing timed out after {max_polls * 2}s (state: {video_file_info.state})")
                raise RuntimeError(f"Video processing timed out after {max_polls * 2}s")

            logger.info("[GEMINI] Video ready. Starting 3-phase metadata pipeline...")

            # ==================== PHASE 1: VIDEO ANALYSIS ====================
            logger.info("[PHASE 1] Analyzing video content with gemini-3.5-flash...")

            phase1_prompt = f"""You are an expert video content analyst. Watch this video CAREFULLY.

The creator says the intended vibe/niche is: {vibe}

Your job is to deeply analyze this video so that a title strategist can craft the perfect viral title.

Focus on:
- The single most striking/funny/shocking moment and WHEN it happens
- The emotional journey a viewer goes through
- Why someone would share this with a friend
- The core irresistible hook
- Specific subjects/entities visible in the video (for SEO)

Be specific and detailed. Reference exact moments in the video."""

            analysis = None
            for model in ["gemini-3.5-flash", "gemini-3.1-flash-lite"]:
                try:
                    analysis = await _call_gemini(
                        client, model,
                        contents=[video_file, phase1_prompt],
                        schema=VideoAnalysis,
                        max_tokens=1500,
                        temperature=0.7,
                    )
                    if analysis:
                        logger.info(f"[PHASE 1] Analysis complete. Core hook: {analysis.get('core_hook', 'N/A')}")
                        break
                except Exception as e:
                    logger.warning(f"[PHASE 1] {model} failed: {e}")
                    continue

            if not analysis:
                raise RuntimeError("Phase 1 (video analysis) failed on all models.")

            # ==================== PHASE 2: TITLE GENERATION ====================
            logger.info("[PHASE 2] Generating title candidates with gemini-3.5-flash...")

            phase2_prompt = f"""You are the #1 YouTube Shorts title strategist. You've generated 50+ viral titles with 10M+ views each.

Here is a detailed analysis of a video in the "{vibe}" niche:

KEY MOMENT: {analysis['key_moment']}
EMOTIONAL ARC: {analysis['emotional_arc']}
SHAREABILITY: {analysis['shareability_factor']}
CORE HOOK: {analysis['core_hook']}
SUBJECTS: {', '.join(analysis['subject_entities'])}

Generate 5 COMPLETELY DIFFERENT title candidates using different hook techniques.
Then rank them and pick the absolute best one.

TITLE RULES:
- UNDER 55 characters (mobile truncation kills reach)
- Authentic lowercase Gen-Z voice — NOT formal English
- Front-load the hook (first 3-4 words must grab attention)
- Create an irresistible curiosity gap or emotional reaction
- NO emojis unless they genuinely add to the vibe (vary it — most titles work better without)
- NEVER use dead patterns: "You won't believe...", "Wait for it...", "Watch until the end"
- NEVER use AI slop: 'unleash', 'epic', 'ultimate', 'revolutionary', 'incredible', 'amazing', 'unbelievable', 'insane'

HALL OF FAME (study these patterns):
- "he wasn't supposed to catch that"
- "bro thought he was safe 💀"
- "this is why nobody invites him"
- "the betrayal at 0:08 though"
- "pov: you finally snapped"
- "tell me this isn't rigged"
- "she did NOT just say that"
- "i can't unsee this"

HALL OF SHAME (NEVER write titles like these):
- "A Funny Dog Playing With A Ball"
- "Amazing Moment Caught On Camera!"
- "You Won't Believe What Happens Next"
- "Epic Fail Caught on Camera"
"""

            best_title = None
            for model in ["gemini-3.5-flash", "gemini-3.1-flash-lite"]:
                for attempt in range(3):
                    try:
                        title_data = await _call_gemini(
                            client, model,
                            contents=[phase2_prompt],
                            schema=TitleCandidates,
                            max_tokens=2000,
                            temperature=1.0,
                            max_attempts=3,  # Let _call_gemini handle API backoffs transparently
                        )
                        if title_data:
                            candidate = title_data.get("best_title", "")
                            quality_issue = _check_title_quality(candidate)
                            if quality_issue:
                                logger.warning(f"[PHASE 2] Title rejected ({model}, attempt {attempt + 1}/3): {quality_issue} — '{candidate}'")
                                # Try picking from the other candidates
                                for alt in title_data.get("candidates", []):
                                    if alt != candidate and not _check_title_quality(alt):
                                        candidate = alt
                                        quality_issue = None
                                        logger.info(f"[PHASE 2] Using alternate candidate: '{candidate}'")
                                        break
                            if not quality_issue:
                                best_title = candidate
                                logger.info(f"[PHASE 2] Winning title: '{best_title}'")
                                logger.info(f"[PHASE 2] All candidates: {title_data.get('candidates', [])}")
                                logger.info(f"[PHASE 2] Reasoning: {title_data.get('ranking_reasoning', 'N/A')}")
                                break
                    except Exception as e:
                        # If we reach here, _call_gemini exhausted its 3 attempts
                        logger.warning(f"[PHASE 2] {model} API completely failed: {e}")
                        break # Stop trying to generate titles with this model and fall back to flash
                if best_title:
                    break

            if not best_title:
                raise RuntimeError("Phase 2 (title generation) failed to produce a quality title.")

            # ==================== PHASE 3: SUPPORTING METADATA ====================
            logger.info("[PHASE 3] Generating description, tags, and engagement metadata with gemini-3.5-flash...")

            phase3_prompt = f"""You are a YouTube Shorts SEO and engagement specialist.

A video in the "{vibe}" niche has been analyzed and titled. Your job is to generate the supporting metadata.

VIDEO ANALYSIS:
- Key moment: {analysis['key_moment']}
- Core hook: {analysis['core_hook']}
- Subjects: {', '.join(analysis['subject_entities'])}

CHOSEN TITLE: "{best_title}"

Generate the description, tags, thumbnail recommendation, and pinned comment that perfectly complement this title.

DESCRIPTION RULES:
- Line 1: Punchy hook or controversial statement (NEVER "In this video..." or "Welcome back")
- Lines 2-3: Natural context with organic SEO keywords
- Final line: 10 to 13 hashtags total, niche-first then mainstream. Vary the exact count and the split
  slightly from video to video instead of always landing on the same numbers — roughly 55-65% should be
  NICHE hashtags tied directly to this video's specific subjects/characters/objects
  (e.g. #SpecificTopic #NicheSubject #TopicVariation), and the remainder MAINSTREAM
  hashtags for broad discovery — the kind of tags real high-view videos in this category actually use
  (e.g. #broadcategory #shorts #relatable #trending — pick whichever genuinely fit this
  video's category/vibe).
- This is a NEW account, so mainstream hashtags are required this time for discovery reach —
  do not skip them or replace them all with niche tags.

TAG RULES (the "Tags" field, separate from hashtags):
- 12 to 15 tags total, niche-first then mainstream. Vary the exact count and the split slightly from
  video to video instead of always landing on the same numbers — roughly 55-65% should be NICHE tags,
  specific multi-word phrases tied to exact subjects: {', '.join(analysis['subject_entities'])}
  (e.g. "specific topic description", "niche action phrase"), and the remainder MAINSTREAM tags —
  broader category/discovery tags real viral videos in this niche rank for (e.g. "broad category",
  "category shorts", "viral humor", "relatable concept" — pick whichever best fit this
  video's category).
- Think "what would someone type into YouTube search to find content LIKE this?" for the mainstream
  portion, and "what would someone type to find THIS EXACT video?" for the niche portion.

PINNED COMMENT: Short, opinionated question that FORCES replies. Under 15 words.
THUMBNAIL: Identify the most dramatic frame with a specific timestamp."""

            metadata = None
            for model in ["gemini-3.5-flash", "gemini-3.1-flash-lite"]:
                try:
                    metadata = await _call_gemini(
                        client, model,
                        contents=[phase3_prompt],
                        schema=SupportingMetadata,
                        max_tokens=1500,
                        temperature=0.8,
                    )
                    if metadata:
                        logger.info(f"[PHASE 3] Metadata complete. Tags: {metadata.get('tags', [])[:5]}...")
                        _log_hashtag_and_tag_counts(
                            metadata.get("description", ""),
                            metadata.get("tags", []),
                            metadata.get("niche_hashtag_count", 0),
                            metadata.get("niche_tag_count", 0),
                        )
                        break
                except Exception as e:
                    logger.warning(f"[PHASE 3] {model} failed: {e}")
                    continue

            if not metadata:
                raise RuntimeError("Phase 3 (supporting metadata) failed on all models.")

            # Combine all phases into final result
            final_result = {
                "title": best_title,
                "description": metadata["description"],
                "tags": metadata["tags"],
                "thumbnail_recommendation": metadata["thumbnail_recommendation"],
                "pinned_comment_suggestion": metadata["pinned_comment_suggestion"],
                # Bonus: include analysis for logging/debugging
                "video_analysis": analysis["key_moment"],
                "target_emotion": analysis["emotional_arc"],
                "hook_style": analysis["core_hook"],
            }

            logger.info(f"[DONE] 3-phase pipeline complete. Title: '{best_title}'")
            return final_result

        finally:
            # Clean up the uploaded file to save quota
            if video_file:
                try:
                    logger.info("[GEMINI] Deleting video from Google's servers...")
                    await client.aio.files.delete(name=video_file.name)
                except Exception as cleanup_err:
                    logger.warning(f"[GEMINI] Failed to delete uploaded video (non-fatal): {cleanup_err}")

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