import asyncio
import json
import logging
import os
import random
import re
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

_LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
_LOG_FILE = os.path.join(_PROJECT_DIR, "youtube_bot.log")

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

if not root_logger.handlers:
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root_logger.addHandler(console_handler)
    try:
        file_handler = RotatingFileHandler(
            _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root_logger.addHandler(file_handler)
    except Exception as e:
        sys.stderr.write(f"Failed to initialize file logging: {e}\n")

logger = logging.getLogger(__name__)


# ── Custom Exceptions ────────────────────────────────────────────────────────

class QuotaExhaustedError(RuntimeError):
    """Raised when Gemini quota is exhausted after all retries on a given model.

    NOTE: quota on the Gemini Developer API free tier is tracked per-model, not
    per-account. Hitting RESOURCE_EXHAUSTED on one model says nothing about the
    remaining quota on another model — they're separate buckets. Callers should
    treat this as "this specific model is tapped out", try the next model in
    the fallback chain, and only treat the situation as "queue for later" once
    every model in the chain has been exhausted.
    """
    pass


# ── Pydantic Schemas ─────────────────────────────────────────────────────────

class VideoDNA(BaseModel):
    """Phase 1 output: Grounded analysis of the video content."""
    video_category: str = Field(
        ...,
        description="Classify as either 'Meme/Relatable/Skit' or 'Normal/Physical/Fails'."
    )
    key_moment: str = Field(
        ...,
        description=(
            "Describe the physical moment on screen (e.g., 'he jerks back and opens his mouth'). "
            "NOTE: If this is a meme, this physical moment is LESS important than the premise."
        )
    )
    most_confusing_detail: str = Field(
        ...,
        description=(
            "Describe the most confusing, unexplained, or bizarre detail in the video. "
            "What behavior or event raises questions?"
        )
    )
    meme_premise: str = Field(
        ...,
        description=(
            "The underlying joke or shared human experience being depicted. "
            "CRITICAL: Do not describe physical movements here. "
            "Bad: 'he jerks back when she pours water'. "
            "Good: 'pretending to sleep because you don't want to get caught awake'."
        )
    )
    relatability_reason: str = Field(
        ...,
        description=(
            "Complete this sentence: 'People relate to this because _____.' "
            "Keep under 12 words. Describe only a common experience. "
            "Do not mention society, psychology, mental health, causes, or theories."
        )
    )
    caption_angle: str = Field(
        ...,
        description=(
            "What would someone text a friend along with this video? "
            "Must be a REACTION to the premise, NOT a description of the video. "
            "Bad: 'he is trying to stay asleep'. Good: 'getting caught awake is somehow worse'."
        )
    )
    reaction_style: str = Field(
        ...,
        description=(
            "The stylistic tone the caption and title should take. "
            "Choose one: 'deadpan', 'relatable', 'confused', or 'opinionated'."
        )
    )
    title_angle: str = Field(
        ...,
        description=(
            "If this clip appeared in a group chat, what is the FIRST thing someone would type? "
            "Requirements: under 10 words, reaction or opinion only, no narration, "
            "no body movements, no explanations, no SEO, no summaries. "
            "It should feel immediately sendable."
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
    click_triggers: list[str] = Field(
        ...,
        description=(
            "Generate exactly 3 title seeds. "
            "For memes: react to the absurdity of the premise or misunderstanding. "
            "For normal videos: point out a specific, visually observable mystery. "
            "The trigger should feel like missing context, making a reader think: 'wait... why?'."
        )
    )


class TitleStudio(BaseModel):
    """Phase 2 output: Multiple title options ranked by quality."""
    candidates: list[str] = Field(
        ...,
        description=(
            "Generate exactly 5 titles. "
            "Be a participant in the joke, not an observer. "
            "Never narrate physical movements (e.g., 'he jerks back', 'he opens his mouth'). "
            "React to the premise/misunderstanding instead. "
            "Do not intentionally force variety. Some titles may end up similar. "
            "Titles should sound slightly lazy, natural, imperfect, and casual "
            "(e.g. sentence case, lowercase, fragments, inconsistent capitalization)."
        )
    )
    ranking_reasoning: str = Field(
        ...,
        description=(
            "Briefly explain why you ranked them in this order. "
            "What makes #1 the most human, deadpan, and casual?"
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
    def cap_title(cls, v: str) -> str:
        if len(v) > 55:
            truncated = v[:52]
            last_space = truncated.rfind(" ")
            if last_space > 20:
                truncated = truncated[:last_space]
            return truncated + "..."
        return v


class PublishingPackage(BaseModel):
    """Phase 3 output: Description, tags, and engagement metadata."""
    description: str = Field(
        ...,
        description=(
            "A casual, conversational YouTube Shorts description. "
            "Act like a participant reacting to the joke, not an observer describing a video. "
            "NEVER describe physical movements (e.g., 'the way he jerks back'). "
            "NEVER explain why it's funny. NEVER psychoanalyze or write mini-essays. "
            "1-2 natural sentences, acting like a text message to a friend reacting to the premise. "
            "Target 200-400 characters before hashtags. "
            "End with 10 to 13 hashtags total on a single line, ordered niche-first then mainstream."
        )
    )
    niche_hashtag_count: int = Field(
        ...,
        ge=0,
        description=(
            "The exact number of hashtags, counting from the START of the hashtag line in `description`, "
            "that are NICHE hashtags (before the mainstream ones begin). This MUST match how you actually "
            "ordered the hashtag line."
        )
    )
    tags: list[str] = Field(
        ...,
        description=(
            "12 to 15 YouTube tags (no # symbols), ordered niche-first then mainstream. "
            "Generate tags from search intents prioritized in this order: "
            "1. Exact scenario, 2. Exact action, 3. Exact subject, 4. Broader category, 5. Adjacent interests. "
            "Every tag must be something a real person would type into search."
        )
    )
    niche_tag_count: int = Field(
        ...,
        ge=0,
        description=(
            "The exact number of tags, counting from the START of the `tags` list, that are NICHE tags "
            "(before the mainstream ones begin)."
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
    def cap_tags(cls, v: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(
            t.strip("#").strip()
            for t in v
            if t.strip("#").strip()
        ))
        if len(cleaned) > 15:
            logger.warning(
                f"[PHASE 3] Model returned {len(cleaned)} tags, truncating to 15. "
                f"Dropped: {cleaned[15:]}"
            )
            cleaned = cleaned[:15]
        if len(cleaned) < 12:
            logger.warning(
                f"[PHASE 3] Model returned only {len(cleaned)} tags, below target range 12-15: {cleaned}"
            )
        return cleaned


# ── Title Quality Gate ───────────────────────────────────────────────────────

_BANNED_TITLE_WORDS = {
    "unleash", "epic", "ultimate", "revolutionary", "incredible", "amazing",
    "unbelievable", "mind-blowing", "jaw-dropping", "insane",
    "#shorts", "subscribe", "like and subscribe", "delve", "dive in",
    "look no further", "mastering", "testament", "revolutionize",
}

_NARRATING_PATTERNS = {
    "this video shows", "in this video", "here's what happens",
    "here is what happens", "this is the moment", "the moment when",
    "the moment where", "when you realize", "this proves that", "this shows",
    "and it's hilarious", "and it's amazing", "so funny", "so wholesome",
    "so satisfying", "this will make you", "guaranteed to", "you won't believe",
    "wait for it", "watch until the end", "must see", "gone wrong", "gone viral",
    "the way he", "he looks so", "she looks so", "the way she",
    "he's really", "she's really", "he is really", "she is really",
}

_RESOLUTION_GIVEAWAYS = {
    "and won", "and lost", "and died", "and survived", "and failed",
    "and succeeded", "and it worked", "and it broke", "ends with",
    "results in", "leads to him", "leads to her",
}


def _check_title_quality(title: str) -> Optional[str]:
    """Returns a rejection reason if the title fails quality checks, None if it passes.

    Ordered by frequency of failure in practice:
      1. Length / formatting
      2. Narrating/explaining (main AI tell)
      3. Giving away the resolution
      4. Banned slop words (last-resort net)
      5. Overly formal capitalization
    """
    if not title or len(title.strip()) < 10:
        return f"Title too short ({len(title.strip())} chars)"

    if len(title.strip()) > 55:
        return f"Title too long ({len(title.strip())} chars)"

    if re.search(r"\b\d{1,2}:\d{2}\b", title):
        return "Contains a timestamp, meaningless on Shorts"

    title_lower = title.lower()

    for pattern in _NARRATING_PATTERNS:
        if pattern in title_lower:
            return f"Narrates/explains instead of hooking: '{pattern}'"

    for pattern in _RESOLUTION_GIVEAWAYS:
        if pattern in title_lower:
            return f"Gives away the resolution: '{pattern}'"

    for banned in _BANNED_TITLE_WORDS:
        if banned in title_lower:
            return f"Contains banned word/phrase: '{banned}'"

    # Flag heavily title-cased titles (e.g. "A Funny Dog Playing With A Ball").
    # - Start from words[1:] — a leading capital is normal in a sentence.
    # - Only count words > 2 chars to avoid flagging "I", "TV", initialisms, etc.
    words = title.split()
    if len(words) >= 6:
        capitalized = sum(1 for w in words[1:] if len(w) > 2 and w[0].isupper())
        if capitalized / (len(words) - 1) > 0.75:
            return "Title looks too formally capitalized (sounds composed, not posted)"

    return None


def _log_hashtag_and_tag_counts(
    description: str,
    tags: list[str],
    niche_hashtag_count: int,
    niche_tag_count: int,
) -> None:
    """Logs hashtag/tag counts and verifies niche/mainstream split. Non-fatal."""
    lines = description.strip().splitlines()
    last_line = lines[-1] if lines else ""
    hashtags = re.findall(r"#\w+", last_line)
    n_hashtags = len(hashtags)
    n_tags = len(tags)

    logger.info(
        f"[PHASE 3] Hashtag count: {n_hashtags} (target 10-13) | "
        f"Tag count: {n_tags} (target 12-15, hard cap 15)"
    )
    if n_hashtags < 10 or n_hashtags > 13:
        logger.warning(f"[PHASE 3] Hashtag count outside target range: got {n_hashtags} — {hashtags}")
    if n_tags < 12:
        logger.warning(f"[PHASE 3] Tag count below target range: got {n_tags} — {tags}")

    if n_hashtags > 0:
        effective_niche = min(niche_hashtag_count, n_hashtags)
        ratio = effective_niche / n_hashtags
        logger.info(f"[PHASE 3] Hashtag niche split: {effective_niche}/{n_hashtags} ({ratio:.0%}, target 55-65%)")
        if not (0.50 <= ratio <= 0.70):
            logger.warning(f"[PHASE 3] Hashtag niche ratio outside range: {effective_niche}/{n_hashtags} = {ratio:.0%}")
        if niche_hashtag_count > n_hashtags:
            logger.warning(f"[PHASE 3] niche_hashtag_count ({niche_hashtag_count}) > actual count ({n_hashtags})")

    if n_tags > 0:
        effective_niche = min(niche_tag_count, n_tags)
        ratio = effective_niche / n_tags
        logger.info(f"[PHASE 3] Tag niche split: {effective_niche}/{n_tags} ({ratio:.0%}, target 55-65%)")
        if not (0.50 <= ratio <= 0.70):
            logger.warning(f"[PHASE 3] Tag niche ratio outside range: {effective_niche}/{n_tags} = {ratio:.0%}")
        if niche_tag_count > n_tags:
            logger.warning(f"[PHASE 3] niche_tag_count ({niche_tag_count}) > actual count ({n_tags})")


def get_gemini_client() -> genai.Client:
    return genai.Client()


async def _call_gemini(
    client,
    model: str,
    contents: list,
    schema,
    max_tokens: int,
    temperature: float,
    sys_instruct: Optional[str] = None,
    max_attempts: int = 3,
):
    """Call Gemini with retries, leveraging native Pydantic parsing and system instructions."""
    for attempt in range(max_attempts):
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruct,
                    response_mime_type="application/json",
                    response_schema=schema,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )

            # 🚀 The SDK handles all the JSON parsing and validation natively
            if response.parsed is not None:
                return response.parsed.model_dump()
            else:
                logger.warning(f"[GEMINI] Empty or unparseable response from {model} on attempt {attempt + 1}/{max_attempts}.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2)

        except APIError as e:
            error_str = str(e).upper()
            is_quota = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            status_code = getattr(e, "code", None)
            is_transient = (
                is_quota or
                "UNAVAILABLE" in error_str or
                "INTERNAL" in error_str or
                "500" in error_str or
                "503" in error_str or
                (isinstance(status_code, int) and status_code >= 500)
            )

            if is_transient and attempt < max_attempts - 1:
                wait_time = 15 if is_quota else 5
                logger.warning(
                    f"[GEMINI] {model} transient API error ({e}) — "
                    f"retrying in {wait_time}s (attempt {attempt + 1}/{max_attempts})..."
                )
                await asyncio.sleep(wait_time)

            elif is_quota:
                logger.error(
                    f"[GEMINI] Quota exhausted on {model} after {max_attempts} attempts. "
                    f"This model's daily limit is likely hit — caller should fall back to "
                    f"another model rather than skip the video outright."
                )
                raise QuotaExhaustedError(f"Gemini quota exhausted on {model} after {max_attempts} retries: {e}") from e

            else:
                raise

        except ValidationError as je:
            # We only need to catch ValidationError now, json.JSONDecodeError is obsolete
            logger.warning(f"[GEMINI] Pydantic validation failed on attempt {attempt + 1}/{max_attempts}: {je}")
            if attempt >= max_attempts - 1:
                raise
            await asyncio.sleep(2)

        except Exception as e:
            logger.warning(f"[GEMINI] Unexpected error on attempt {attempt + 1}/{max_attempts}: {e}")
            if attempt >= max_attempts - 1:
                raise
            await asyncio.sleep(2)

    return None


async def generate_metadata_async(video_path: str, content_brief: str = "") -> Optional[dict]:
    """
    3-phase metadata generation pipeline:
      Phase 1 (Analysis):  model watches video → deep analysis + 3 title-seed observations
      Phase 2 (Title):     model WATCHES VIDEO AGAIN + uses seeds → 5 candidates → picks best
      Phase 3 (Metadata):  generates description, tags, thumbnail rec, pinned comment

    Returns None on quota exhaustion (caller should queue for retry) or None on unrecoverable
    failure (caller should log and skip). Raises RuntimeError for unexpected errors.
    """
    client = get_gemini_client()
    video_file = None

    brief_cleaned = content_brief.strip()[:500]
    brief_block = ""
    if brief_cleaned:
        brief_block = f"""

CREATOR CONTEXT (optional guidance):
{brief_cleaned}

Use this only to guide framing and tone.
The video is the source of truth.
Never invent details that are not visible."""

    # Gemini's free-tier quota is tracked per-model, not per-account, so exhausting
    # gemini-3.5-flash says nothing about gemini-3.1-flash-lite's remaining quota.
    # Every phase below tries each model in order and only treats the situation as
    # "fully exhausted, queue this video for later" once every model in this list
    # has individually hit RESOURCE_EXHAUSTED.
    _FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]
    logger.info(f"Gemini fallback model chain: {_FALLBACK_MODELS}")

    logger.info(f"[GEMINI] Uploading video: {video_path}")
    try:
        video_file = await client.aio.files.upload(file=video_path)
        logger.info("[GEMINI] Uploaded. Waiting for processing...")

        try:
            max_polls = 60
            video_file_info = None
            for _ in range(max_polls):
                video_file_info = await client.aio.files.get(name=video_file.name)
                if video_file_info.state == "ACTIVE":
                    break
                elif video_file_info.state == "FAILED":
                    raise RuntimeError("Video processing failed on Google's end.")
                await asyncio.sleep(2)
            else:
                raise RuntimeError(f"Video processing timed out after {max_polls * 2}s (state: {video_file_info.state})")

            logger.info("[GEMINI] Video ready. Starting 3-phase pipeline...")

            # ════════════════════ PHASE 1: VIDEO ANALYSIS ════════════════════
            logger.info("[PHASE 1] Analyzing video...")

            phase1_prompt = f"""You are not a critic.
You are not a psychologist.
You are not a marketer.
You are not a storyteller.

You are watching a short clip to casually post it online.{brief_block}

CRITICAL DIFFERENCE - DETERMINE THE CATEGORY:
Is this a literal video (animals doing things, cooking, fails) OR a Meme/Skit/Relatable Post?

If it is a MEME, SKIT, or RELATABLE POST:
- Identify the JOKE, MISUNDERSTANDING, or SHARED EXPERIENCE. 
- DO NOT focus on the physical movements. 
- Example Meme: "When your mom pours water on you but you can't react because you're pretending to sleep."
- Bad Analysis: "He jerks back and opens his mouth." (This is physical narration).
- Good Analysis: "Pretending to sleep because you don't want to get caught awake." (This is the premise).

If it is a NORMAL VIDEO (Fail, Animal, Satisfying):
- Analyze literal physical events and visual context.

Never infer:
- diagnoses
- motivations
- character identities not shown
- causes
- psychological states

Focus on:
- Video Category (video_category)
- The overall tone (reaction_style)
- The obvious joke or premise (meme_premise)
- A casual, natural caption suggestion (caption_angle)
- Specific subjects/entities visible in the video (subject_entities)
- Exactly 3 title seeds / click triggers (click_triggers). 
  For memes: react to the absurdity. 
  For normal videos: point out a specific, visually observable mystery.

FINAL GROUNDING CHECK:
Could someone point at the video and say "yes, that's in there" or "yes, that's the obvious joke"? If not, do not include it."""

            analysis = None
            quota_exhausted_models = 0
            for model in _FALLBACK_MODELS:
                try:
                    analysis = await _call_gemini(
                        client, model,
                        contents=[video_file],
                        schema=VideoDNA,
                        max_tokens=1500,
                        temperature=0.3,
                        sys_instruct=phase1_prompt,
                    )
                    if analysis:
                        logger.info(f"[PHASE 1] Done. Category: {analysis.get('video_category', 'Unknown')}")
                        logger.info(f"[PHASE 1] Meme premise: {analysis.get('meme_premise', 'N/A')}")
                        logger.info(
                            "[PHASE 1] Confusing: %s | Relatability: %s | Caption Angle: %s | Title Angle: %s",
                            analysis.get("most_confusing_detail"),
                            analysis.get("relatability_reason"),
                            analysis.get("caption_angle"),
                            analysis.get("title_angle"),
                        )
                        logger.info(f"[PHASE 1] Title seeds: {analysis.get('click_triggers', [])}")
                        break
                except QuotaExhaustedError:
                    logger.warning(f"[PHASE 1] {model} quota exhausted — falling back to next model...")
                    quota_exhausted_models += 1
                    continue
                except Exception as e:
                    logger.warning(f"[PHASE 1] {model} failed: {e}", exc_info=True)
                    continue

            if not analysis:
                if quota_exhausted_models == len(_FALLBACK_MODELS):
                    raise QuotaExhaustedError("All models exhausted their quota in Phase 1 (video analysis).")
                raise RuntimeError("Phase 1 (video analysis) failed on all models.")

            # ════════════════════ PHASE 2: TITLE GENERATION ════════════════════
            logger.info("[PHASE 2] Generating title candidates...")

            is_meme = "Meme" in analysis.get("video_category", "")

            if is_meme:
                phase2_context_block = f"""
VIDEO CATEGORY: {analysis.get('video_category')}
REACTION STYLE: {analysis.get('reaction_style', 'deadpan')}
MEME PREMISE: {analysis['meme_premise']}
CAPTION ANGLE: {analysis.get('caption_angle', '')}
TITLE ANGLE: {analysis.get('title_angle', '')}
"""
            else:
                seeds_block = "\n".join(f"  • {s}" for s in analysis.get("click_triggers", []))
                phase2_context_block = f"""
VIDEO CATEGORY: {analysis.get('video_category')}
REACTION STYLE: {analysis.get('reaction_style', 'deadpan')}
KEY MOMENT: {analysis['key_moment']}
RAW TITLE SEEDS:
{seeds_block}
"""

            phase2_prompt = f"""You are a person who just watched a clip and is casually texting it to a friend.{brief_block}

ROLE: PARTICIPANT, NOT OBSERVER.
You are participating in the joke. You are NOT describing the video.

{phase2_context_block.strip()}

IF THIS IS A MEME/SKIT/RELATABLE POST:
- The MEME PREMISE is the only thing that matters.
- Never describe body movements, facial expressions, or play-by-play actions.
- React to the absurdity, misunderstanding, or shared experience.

Examples of GOOD meme titles (Reactions):
- pretending to be asleep got way too serious
- bro refused to break character 😭
- getting caught awake is somehow worse
- fake sleeping around your parents is different

Examples of BAD meme titles (Observations):
- he's really trying to stay asleep
- he looks terrified
- the way he opens his mouth
- bro was not okay

ALL TITLES MUST BE:
- Under 55 characters — hard limit.
- Casual and natural. Sentence case, lowercase, and fragments are encouraged.
- No narration ("watch as", "this video shows", "the moment when").

Do not rush. Return only the strongest 5."""

            best_title = None
            last_title_data = None  # kept for quality-gate fallback
            quota_exhausted_models = 0

            for model in _FALLBACK_MODELS:
                try:
                    title_data = await _call_gemini(
                        client, model,
                        contents=[video_file],
                        schema=TitleStudio,
                        max_tokens=2000,
                        temperature=0.9,
                        sys_instruct=phase2_prompt,
                        max_attempts=3,
                    )
                    if title_data:
                        last_title_data = title_data
                        candidate = title_data.get("best_title", "")
                        quality_issue = _check_title_quality(candidate)
                        initial_candidate = candidate
                        initial_issue = quality_issue

                        if quality_issue:
                            # Scan alternates before re-prompting or moving to next model
                            for alt in title_data.get("candidates", []):
                                if alt != candidate and not _check_title_quality(alt):
                                    candidate = alt
                                    quality_issue = None
                                    break

                        if not quality_issue:
                            best_title = candidate
                            logger.info(f"[PHASE 2] Candidates: {title_data.get('candidates', [])}")
                            logger.info(f"[PHASE 2] Winner: {best_title}")
                            if initial_issue:
                                logger.info(f"[PHASE 2] Rejected: '{initial_candidate}' (Reason: {initial_issue})")
                            logger.info(f"[PHASE 2] Reasoning: {title_data.get('ranking_reasoning', 'N/A')}")
                            break
                        else:
                            logger.warning(
                                f"[PHASE 2] {model} candidates failed quality checks. "
                                f"Candidates: {title_data.get('candidates', [])}. "
                                f"Rejected best_title: '{initial_candidate}' (Reason: {initial_issue})"
                            )

                except QuotaExhaustedError:
                    logger.warning(f"[PHASE 2] {model} quota exhausted — falling back to next model...")
                    quota_exhausted_models += 1
                    continue
                except Exception as e:
                    logger.warning(f"[PHASE 2] {model} failed: {e}", exc_info=True)
                    continue

            # Quality-gate fallback: a slightly imperfect title is always better than
            # crashing the pipeline and skipping the upload entirely.
            if not best_title and last_title_data:
                fallback = last_title_data.get("best_title", "")
                if fallback:
                    best_title = fallback
                    logger.warning(
                        f"[PHASE 2] All quality checks failed — using raw best_title as fallback: '{fallback}'"
                    )
                    logger.info(f"[PHASE 2] Candidates: {last_title_data.get('candidates', [])}")
                    logger.info(f"[PHASE 2] Winner: {best_title}")
                    logger.info(f"[PHASE 2] Rejected: all candidates failed quality checks, used raw best_title as fallback")

            if not best_title:
                if quota_exhausted_models == len(_FALLBACK_MODELS):
                    raise QuotaExhaustedError("All models exhausted their quota in Phase 2 (title generation).")
                raise RuntimeError("Phase 2 (title generation) failed to produce any title.")

            # ════════════════════ PHASE 3: SUPPORTING METADATA ════════════════════
            logger.info("[PHASE 3] Generating description, tags, and engagement metadata...")

            if is_meme:
                phase3_context_block = f"""
VIDEO CATEGORY: {analysis.get('video_category')}
REACTION STYLE: {analysis.get('reaction_style', 'deadpan')}
MEME PREMISE: {analysis['meme_premise']}
RELATABILITY REASON: {analysis['relatability_reason']}
CAPTION ANGLE: {analysis.get('caption_angle', '')}
TITLE ANGLE: {analysis.get('title_angle', '')}
"""
            else:
                phase3_context_block = f"""
VIDEO CATEGORY: {analysis.get('video_category')}
REACTION STYLE: {analysis.get('reaction_style', 'deadpan')}
KEY MOMENT: {analysis['key_moment']}
SUBJECTS: {', '.join(analysis['subject_entities'])}
"""

            phase3_prompt = f"""You are a person texting a friend about a clip you just sent them.{brief_block}

{phase3_context_block.strip()}

CHOSEN TITLE: "{best_title}"

DESCRIPTION RULES:
- Write 1-2 natural sentences. 
- Prefer 1 medium sentence OR 2 short-medium sentences. Do not elaborate after the main thought is complete.
- Fragments are encouraged. Complete sentences are not required.
- BE A PARTICIPANT. React to the joke or premise using the REACTION STYLE.
- DO NOT act like a movie reviewer, commentator, or AI.
- NEVER describe physical movements on screen (e.g., "The way he jerks back...", "He opens his mouth...").
- NEVER explain why the joke works.
- NEVER psychoanalyze, invent backstories, or write mini-essays about "life lessons" or "mental health".

STRUCTURE:
Just a natural, single thought reacting to the premise. 
Think: "What would I type in a text message if I sent this meme to the group chat?"

Examples of GOOD descriptions:
• fake sleeping around your parents becomes serious business.
• getting caught awake somehow feels worse.
• i know exactly why he refused to break character.
• waking up at 6am doesn't count if you're still awake.

Examples of BAD descriptions (DO NOT DO THIS):
• The way he jerks back and opens his mouth is actually terrifying because he is fully committed to the bit.
• This video shows how sleep deprivation causes hallucinations...
• Watch this hilarious moment where...

HASHTAG RULES:
- 10 to 13 hashtags total, niche-first then mainstream. Placed at the end on a single line.

TAG RULES:
- 12 to 15 tags, niche-first then mainstream.

PINNED COMMENT: Short opinionated question that FORCES replies. Under 15 words.

Before returning the description, privately run this self-critique:
1. Did I describe physical body movements? (If yes, discard)
2. Does this sound like an essay or explanation? (If yes, discard)
3. Would a real person type this in a group chat? (If no, discard)"""

            metadata = None
            quota_exhausted_models = 0
            for model in _FALLBACK_MODELS:
                try:
                    metadata = await _call_gemini(
                        client, model,
                        contents=[video_file, "Produce the PublishingPackage."],
                        schema=PublishingPackage,
                        max_tokens=1500,
                        temperature=0.6,
                        sys_instruct=phase3_prompt,
                    )
                    if metadata:
                        logger.info(f"[PHASE 3] Done. Tags sample: {metadata.get('tags', [])[:5]}...")
                        _log_hashtag_and_tag_counts(
                            metadata.get("description", ""),
                            metadata.get("tags", []),
                            metadata.get("niche_hashtag_count", 0),
                            metadata.get("niche_tag_count", 0),
                        )
                        break
                except QuotaExhaustedError:
                    logger.warning(f"[PHASE 3] {model} quota exhausted — falling back to next model...")
                    quota_exhausted_models += 1
                    continue
                except Exception as e:
                    logger.warning(f"[PHASE 3] {model} failed: {e}", exc_info=True)
                    continue

            if not metadata:
                if quota_exhausted_models == len(_FALLBACK_MODELS):
                    raise QuotaExhaustedError("All models exhausted their quota in Phase 3 (supporting metadata).")
                raise RuntimeError("Phase 3 (supporting metadata) failed on all models.")

            final_result = {
                "title": best_title,
                "description": metadata["description"],
                "tags": metadata["tags"],
                "thumbnail_recommendation": metadata["thumbnail_recommendation"],
                "pinned_comment_suggestion": metadata["pinned_comment_suggestion"],
                "video_category": analysis.get("video_category", "Unknown"),
                "video_analysis": analysis["key_moment"],
                "meme_premise": analysis["meme_premise"],
                "relatability_reason": analysis["relatability_reason"],
                "caption_angle": analysis.get("caption_angle", ""),
                "title_angle": analysis.get("title_angle", ""),
            }

            logger.info(f"[DONE] Pipeline complete. Title: '{best_title}'")
            return final_result

        finally:
            if video_file:
                try:
                    logger.info("[GEMINI] Deleting uploaded video from Google's servers...")
                    await client.aio.files.delete(name=video_file.name)
                except Exception as cleanup_err:
                    logger.warning(f"[GEMINI] Failed to delete uploaded video (non-fatal): {cleanup_err}")

    except QuotaExhaustedError as e:
        logger.error(
            f"[QUOTA] All fallback models exhausted their Gemini quota — skipping '{video_path}'. "
            f"Queue for retry when quota resets. Error: {e}"
        )
        return None

    except APIError as e:
        logger.error(f"Gemini API error: {e}")
        raise RuntimeError(f"Gemini API error: {e}") from e

    except Exception as e:
        logger.error(f"Unexpected error generating metadata: {e}")
        raise RuntimeError(f"Unexpected error generating metadata: {e}") from e


# ── Quick Test ───────────────────────────────────────────────────────────────

async def _test():
    brief = (
        "Funny animal meme. The joke is that the dog has an irrational fear of "
        "a kitchen drawer and freezes whenever it opens. Titles should feel like "
        "a friend texting another friend after seeing something weird. Avoid generic "
        "clickbait and reaction memes."
    )
    result = await generate_metadata_async(
        video_path="videos/meme1.mp4",
        content_brief=brief,
    )
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("Metadata generation returned None — quota exhausted or upload failed. Check logs.")


if __name__ == "__main__":
    asyncio.run(_test())