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
    key_moment: str = Field(
        ...,
        description=(
            "Describe the single most striking, funny, shocking, or satisfying moment in the video. "
            "Be specific — reference what happens visually and when it occurs."
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
            "State the situation being depicted. Use one sentence. "
            "Only use actions, visible text, or obvious misunderstandings. "
            "Never use diagnoses, explanations, reasons, interpretations, or psychology."
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
            "Each seed must be: one specific moment, visually observable, "
            "impossible to understand without seeing the clip, and useful for creating curiosity. "
            "The trigger should feel like missing context, making a reader think: 'wait... why?'. "
            "Bad: 'dog barks at fridge'. Good: 'the dog freezes every time this drawer opens'. "
            "Bad: 'person gets surprised'. Good: 'he keeps checking under the couch for something'."
        )
    )


class TitleStudio(BaseModel):
    """Phase 2 output: Multiple title options ranked by quality."""
    candidates: list[str] = Field(
        ...,
        description=(
            "Generate exactly 5 titles. "
            "Write titles the way someone would text a friend after seeing this clip. "
            "Pretend your friend can only see the title and a thumbnail. "
            "Do not intentionally force variety. Some titles may end up similar. That is acceptable. "
            "A title should feel like a simple observation, reaction, opinion, or question. "
            "Never summarize, explain, market, optimize, or narrate. "
            "Titles should sound slightly lazy, natural, and imperfect (e.g. sentence case, lowercase, fragments, inconsistent capitalization are all acceptable). "
            "Discard any candidate that is generic, sounds like marketing, reveals the ending/resolution, or sounds like an AI generator."
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
            "A casual, specific YouTube Shorts description. Write the description like a real person talking about the clip after posting it. "
            "The description should be 1-3 natural sentences (prefer two medium sentences or three short ones) and target 200-400 characters before hashtags. "
            "May include one additional observation, a reaction/opinion about the joke, brief context visible in the video, or a common experience directly implied by the clip. "
            "Stop once the thought feels complete. "
            "End with 10 to 13 hashtags total on a single line, ordered niche-first then mainstream. "
            "Generate hashtags strictly from specific subjects, central/meme premise, and content category. "
            "Never invent off-screen info, write stories/backstories, diagnose people, explain hidden meanings, or use marketing language."
        )
    )
    niche_hashtag_count: int = Field(
        ...,
        ge=0,
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
            "Generate tags from search intents prioritized in this order: "
            "1. Exact scenario, "
            "2. Exact action, "
            "3. Exact subject, "
            "4. Broader category, "
            "5. Adjacent interests. "
            "Prefer real search phrases over keywords (e.g. 'dog afraid of microwave' instead of 'dog' or 'funny'). "
            "Every tag must be something a real person would type into search (if you would be surprised to see the phrase in YouTube autocomplete, discard it). "
            "Avoid tags that sound like metadata (e.g. 'funny canine interaction', 'unexpected response'). "
            "If it could be copied onto another upload with minimal changes, reject it."
        )
    )
    niche_tag_count: int = Field(
        ...,
        ge=0,
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

You are a person who just watched a short clip and is casually posting it.

Watch this video CAREFULLY.{brief_block}

Analyze this video literally.

Only infer:
1. what is visible
2. what text explicitly says
3. the obvious joke or premise

Never infer:
- diagnoses
- motivations
- backstories
- character identities not shown
- causes
- symbolism
- references
- psychological states

Focus on:
- The single most striking/funny/shocking moment and exactly when it happens (key_moment)
- The most confusing, unexplained, or bizarre detail in the video (most_confusing_detail)
- If the video is a meme, the literal premise/joke of the meme (meme_premise)
- The literal relatability reason (relatability_reason)
- Specific subjects/entities visible in the video (subject_entities)
- Exactly 3 title seeds / click triggers (click_triggers). Each seed must be:
  • one specific moment
  • visually observable
  • impossible to understand without seeing the clip
  • useful for creating curiosity (the trigger should feel like missing context, making the reader think "wait... why?")
  
  Bad: dog barks at fridge
  Good: the dog freezes every time this drawer opens
  Bad: person gets surprised
  Good: he keeps checking under the couch for something

Be specific and visual. Reference exact moments in the video.

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
                        logger.info(f"[PHASE 1] Done. Meme premise: {analysis.get('meme_premise', 'N/A')}")
                        logger.info(
                            "[PHASE 1] Confusing: %s | Relatability: %s",
                            analysis.get("most_confusing_detail"),
                            analysis.get("relatability_reason"),
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

            seeds_block = "\n".join(f"  • {s}" for s in analysis.get("click_triggers", []))

            phase2_prompt = f"""You are not a critic.
You are not a psychologist.
You are not a marketer.
You are not a storyteller.

You are a person who just watched a short clip and is casually posting it.

Write titles the way someone would text a friend after seeing this clip.{brief_block}

A title should feel like:
- an observation
- a reaction
- an opinion
- a question

Never:
- summarize
- explain
- market
- optimize
- narrate

Titles should sound slightly lazy and imperfect.

---

Examples of GOOD titles:
- the dog looked at him for way too long
- bro brought a ladder for this
- nobody noticed the kid in the background
- he keeps checking the same drawer
- the cashier's face at the end 😭

Examples of BAD titles:
- you won't believe what happened
- funniest video ever
- this is so satisfying
- wait for it
- watch until the end
- incredible moment caught on camera

---

Generate five titles.
Do not force them to be different.
Generate the five titles you genuinely think a real uploader might use.
Some titles may end up similar. That is acceptable.

---

Everything in this title must come from what's actually visible in THIS specific video.

VIDEO ANALYSIS:
KEY MOMENT: {analysis['key_moment']}
MEME PREMISE: {analysis['meme_premise']}

RAW TITLE SEEDS (specific observations from the analyst — use these as your starting material):
{seeds_block}

A reader should feel: "wait, why?"
Do not intentionally engineer curiosity. Simply describe an incomplete situation.
State the SETUP only. The outcome, punchline, and twist must not appear in the title.

ALL OTHER RULES (every candidate must satisfy all of these):
- Under 55 characters — hard limit, mobile truncates here
- Capitalization should feel natural. Sentence case, lowercase, fragments, and inconsistent capitalization are all acceptable if they fit the clip.
- No emotional labels ("hilarious", "wholesome", "satisfying") — show the thing, not the feeling
- No narration ("watch as", "this video shows", "the moment when") — drop the viewer in directly
- No emojis unless one specific emoji is doing real comedic work (most titles don't need one)
- Fragments, slightly wrong grammar, weird specificity — all acceptable, all real

FINAL GROUNDING CHECK:
Could someone point at the video and say "yes, that's in there" or "yes, that's the obvious joke"? If not, do not include it.

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
            # crashing the pipeline and skipping the upload entirely. Log it loudly so
            # you can tune the gate or the prompts later.
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

            phase3_prompt = f"""You are not a critic.
You are not a psychologist.
You are not a marketer.
You are not a storyteller.

You are a person who just watched a short clip and is casually posting it.
The description should feel like something someone typed in under a minute, not something they spent 10 minutes composing.

You are writing the description for ONE YouTube Short.
Write it the way an actual person posting THIS specific video would — casual and specific.{brief_block}

VIDEO ANALYSIS:
- Key moment: {analysis['key_moment']}
- Meme premise: {analysis['meme_premise']}
- Relatability reason: {analysis['relatability_reason']}
- Subjects: {', '.join(analysis['subject_entities'])}

CHOSEN TITLE: "{best_title}"

DESCRIPTION LENGTH:
- 1-3 sentences
- Target 200-400 characters before hashtags. Do not add filler just to reach the target length.
- Prefer two medium sentences or three short ones.
- Stop once the thought feels complete

The description should be 1-3 natural sentences and around 200-400 characters before hashtags.
Write it like a real person talking about the clip after posting it.

PREFERRED STRUCTURE:
Sentence 1:
specific observation from the clip.
Sentence 2:
reaction, opinion, or common experience directly implied by the clip.
Sentence 3 (optional):
one final deadpan comment.
Stop.

The description may include:
- one additional observation
- a reaction or opinion about the joke
- brief context directly visible in the video or explicitly stated in on-screen text
- a common experience directly implied by the clip

Naturally mention important subjects and themes from the clip when relevant.

Never:
- invent off-screen information
- create lore or backstories
- diagnose people
- explain hidden meanings
- use marketing language
- write generic engagement bait

The description should read like a friend talking about the clip, not like SEO copy.

UNIQUENESS CHECK:
If this description could fit more than 20% of YouTube Shorts, rewrite it.
Mention at least one specific detail from THIS clip.

Examples of GOOD description lines:
• I thought he had finally fixed his sleep schedule until the text said he was actually going to bed at 6 AM. The parents celebrating somehow makes this even worse because they're genuinely proud of him.

• The dog freezes every single time that drawer opens and nobody else in the room seems concerned about it anymore. At some point they probably just accepted that this is his enemy.

• I thought this was about to be another sad breakup edit and then the last line completely changed the mood. The switch happens so fast that your brain barely catches up.

Examples of BAD description lines:
• watch this hilarious dog video
• what do you think?
• like and subscribe
• in this video...
• wait until the end

Before returning the description, privately run this self-critique rubric and discard any that fail:
1. Did I restate the title? (If yes, discard)
2. Did I solve the mystery? (If yes, discard)
3. Does this sound like a YouTube template? (If yes, discard)
4. Would a real person type this? (If no, discard)
5. Did I follow the sentence length (1-3 sentences, 200-400 chars)? (If no, discard)

HASHTAG RULES (placed at the end on a single line, same field):
- 10 to 13 hashtags total, niche-first then mainstream. Vary count and split slightly each time.
- Generate hashtags strictly from: subjects, central/meme premise, and content category.
- Discovery labels, not SEO stuffing. If removing a hashtag would not improve discoverability, do not include it (avoid filler like #omg, #cool, #lol, #awesome).
- Never use banned hashtags: #fyp, #foryou, #viral, #xyzbca, #explorepage, #blowup.

TAG RULES ("tags" field, separate from hashtags):
- 12 to 15 tags, niche-first then mainstream. Vary count and split slightly each time.
- Generate tags from search intents prioritized in this order:
  1. Exact scenario
  2. Exact action
  3. Exact subject
  4. Broader category
  5. Adjacent interests
- Prefer real search phrases over keywords (e.g. "dog afraid of microwave", "funny dog behavior" instead of "dog", "funny").
- Every tag must be something a real person would type into search (if you would be surprised to see this exact phrase in YouTube autocomplete, discard it).
- Avoid tags that sound like metadata instead of a search query (e.g. "funny canine interaction", "unexpected response"). If it could be copied onto another upload with minimal changes, reject it.

PINNED COMMENT: Short opinionated question that FORCES replies. Under 15 words.
Sound like a nosy person stirring something, not a survey.

THUMBNAIL: Most dramatic frame, specific timestamp, overlay suggestion for max CTR.

FINAL GROUNDING CHECK:
For every title, description, hashtag, and tag:
Could someone point at the video and say "yes, that's in there" or "yes, that's the obvious joke"? If not, remove it/do not include it.

Every sentence in the description must pass this test:
1. Could someone verify this by watching the clip?
OR
2. Is this an obvious reaction to the joke?
If neither is true, remove the sentence."""

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
                "video_analysis": analysis["key_moment"],
                "meme_premise": analysis["meme_premise"],
                "relatability_reason": analysis["relatability_reason"],
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
        # Return None so the caller can queue this video for later rather than crashing.
        # By the time we get here, every model in _FALLBACK_MODELS has been tried and
        # exhausted its own quota — this is a real "come back later" situation, not
        # just one model having a bad day.
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