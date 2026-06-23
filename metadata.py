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
            "Imagine you watched this clip and immediately sent it to a friend. "
            "What would you type when sending this clip to one friend? That's the title. "
            "Pretend your friend can only see the title and a thumbnail. "
            "The title's job is to make them ask for the missing context. "
            "All 5 titles MUST feel meaningfully different (written by five different people, not simply rewording the same idea). "
            "Each title must write from a different perspective: "
            "1. someone confused by the clip "
            "2. someone who sees themselves in the situation "
            "3. someone who noticed a weird detail "
            "4. someone worried about what happens next "
            "5. someone disagreeing with what's happening. "
            "Capitalization should feel natural. Sentence case, lowercase, fragments, and inconsistent capitalization "
            "are all acceptable if they fit the clip. "
            "Bad titles: explain, summarize, advertise, use generic templates, or could belong to another video. "
            "Avoid defaulting to reaction memes, Twitter captions, or Reddit-style comments (e.g., 'bro really said', 'nah because why', 'i'm crying 😭'). "
            "Prefer describing a specific thing that happened in the clip. "
            "For each candidate, privately run this self-critique rubric: "
            "1. Could this fit 1000 other videos? "
            "2. Does it reveal the ending? "
            "3. Does it sound like marketing? "
            "4. Does it mention a specific visual detail? "
            "5. Would I genuinely click this? "
            "6. If replacing one noun makes the title fit another video, is it too generic? "
            "Discard any candidate that fails. If it sounds like it came from an AI title generator, discard it immediately."
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
            "A casual, specific YouTube Shorts description. "
            "Write the description the way someone would caption this clip after posting it (e.g. an observation, opinion, reaction, extra context, mini-story, or deadpan statement — do not default to questions). "
            "The description should feel like the second thing someone says after showing a friend the clip (the next sentence in the conversation). "
            "It should feel like it was typed in under 10 seconds; fragments, lowercase, and incomplete thoughts are acceptable. "
            "The description's job is to reward curiosity created by the title by adding exactly one new piece of information without restating the title, explaining everything, or solving the mystery. "
            "End with 10 to 13 hashtags total on a single line, ordered niche-first then mainstream. "
            "Generate hashtags prioritizing: exact subjects in the video -> exact category of content -> broad discovery hashtags. "
            "If removing a hashtag would not improve discoverability, do not include it (avoid filler like #omg, #cool, #lol, #awesome). "
            "Never use phrasing like 'this clip', 'this video', 'this one', 'caught on camera', generic AI intros, or generic engagement questions. "
            "Before returning the description, privately run this self-critique rubric: "
            "1. Did I restate the title? "
            "2. Did I solve the mystery? "
            "3. Does this sound like a YouTube template? "
            "4. Would a real person type this? "
            "5. Did I add exactly one new piece of information? "
            "If it could be copied onto another upload with minimal changes, reject it."
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
            is_unavailable = "UNAVAILABLE" in error_str

            if (is_quota or is_unavailable) and attempt < max_attempts - 1:
                wait_time = 15 if is_quota else 5
                logger.warning(
                    f"[GEMINI] {model} {'rate-limited' if is_quota else 'unavailable'} — "
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


async def generate_metadata_async(video_path: str, vibe: str) -> Optional[dict]:
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

            phase1_prompt = f"""You are an expert video content analyst. Watch this video CAREFULLY.

The creator's intended vibe/niche: {vibe}

Deeply analyze this video so a title strategist can craft the perfect viral title.

Focus on:
- The single most striking/funny/shocking moment and exactly when it happens
- The emotional journey a viewer goes through start to finish
- Why someone would send this to a friend
- The core irresistible hook that makes it impossible to scroll past
- Specific subjects/entities visible in the video (for SEO)
- Exactly 3 title seeds (click triggers). Each seed must be:
  • one specific moment
  • visually observable
  • impossible to understand without seeing the clip
  • useful for creating curiosity (the trigger should feel like missing context, making the reader think "wait... why?")
  
  Bad: dog barks at fridge
  Good: the dog freezes every time this drawer opens
  Bad: person gets surprised
  Good: he keeps checking under the couch for something

Be specific and visual. Reference exact moments in the video."""

            analysis = None
            quota_exhausted_models = 0
            for model in _FALLBACK_MODELS:
                try:
                    analysis = await _call_gemini(
                        client, model,
                        contents=[video_file],
                        schema=VideoDNA,
                        max_tokens=1500,
                        temperature=0.7,
                        sys_instruct=phase1_prompt,
                    )
                    if analysis:
                        logger.info(f"[PHASE 1] Done. Core hook: {analysis.get('core_hook', 'N/A')}")
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

            # Techniques that reliably produce grounded, specific titles.
            # Removed "incomplete comparison" (produces fragment garbage like "funnier than—")
            # and "deadpan label" (produces corporate-speak like "individual disrupts equilibrium").
            # Both fail because they're abstract methods that don't anchor to specific content.
            _TITLE_PERSPECTIVES = [
                "someone confused by the clip",
                "someone who sees themselves in the situation",
                "someone who noticed a weird detail",
                "someone worried about what happens next",
                "someone disagreeing with what's happening",
            ]
            perspectives_block = "\n".join(f"- {p}" for p in _TITLE_PERSPECTIVES)
            seeds_block = "\n".join(f"  • {s}" for s in analysis.get("click_triggers", []))

            phase2_prompt = f"""Imagine you watched this clip and immediately sent it to a friend.

What would you type when sending this clip to one friend? That's the title.

Pretend your friend can only see the title and a thumbnail.
The title's job is to make them ask for the missing context.

Do not optimize. Do not advertise. Do not summarize. Just make someone need to click.

---

Guiding principle:

The more specific the setup,
and the less complete the explanation,
the stronger the title usually is.

Titles should feel discovered, not manufactured.

The viewer should think:
"why did that happen?"
"What am I missing?"
"there's definitely context here."

Never answer those questions in the title.

---

Examples of GOOD titles:
- the dog looked at him for way too long
- bro brought a ladder for this
- nobody noticed the kid in the background
- he keeps checking the same drawer
- the cashier's face at the end 😭

Why they're good:
- specific
- incomplete
- imply a story
- could only belong to one kind of clip

Examples of BAD titles:
- you won't believe what happened
- funniest video ever
- this is so satisfying
- wait for it
- watch until the end
- incredible moment caught on camera

Bad titles:
- explain
- summarize
- advertise
- use generic templates
- could belong to another video

Reject anything that does these things.

Avoid defaulting to reaction memes, Twitter captions, or Reddit-style comments (e.g., "bro really said", "nah because why", "i'm crying 😭"). Prefer describing a specific thing that happened in the clip.

---

All 5 titles MUST feel meaningfully different.
A reader should believe they were written by five different people.
Do not simply reword the same idea.

For each candidate, privately run this self-critique rubric and discard any that fail:
1. Could this fit 1000 other videos? (If yes, discard)
2. Does it reveal the ending? (If yes, discard)
3. Does it sound like marketing? (If yes, discard)
4. Does it mention a specific visual detail? (If no, discard)
5. Would I genuinely click this? (If no, discard)
6. Does it sound like it came from an AI title generator? (If yes, discard immediately)
7. If replacing one noun makes the title fit another video, is it too generic? (If yes, discard. e.g., "the dog did something weird" -> "the cat did something weird" still works. Whereas "the dog freezes every time the microwave beeps" falls apart if you swap nouns.)

---

Everything in this title must come from what's actually visible in THIS specific video.

VIDEO ANALYSIS:
KEY MOMENT: {analysis['key_moment']}
EMOTIONAL ARC: {analysis['emotional_arc']}
SHAREABILITY: {analysis['shareability_factor']}
CORE HOOK: {analysis['core_hook']}
SUBJECTS: {', '.join(analysis['subject_entities'])}

RAW TITLE SEEDS (specific observations from the analyst — use these as your starting material):
{seeds_block}

For each of the 5 candidate titles, write from one of the following natural human perspectives/reactions:
{perspectives_block}

THE ONLY RULE THAT MATTERS: a title must CREATE a gap, not CLOSE one.
If a viewer reads it and already knows how the video ends — you've failed.
State the SETUP only. The outcome, punchline, and twist must not appear in the title.

ALL OTHER RULES (every candidate must satisfy all of these):
- Under 55 characters — hard limit, mobile truncates here
- Capitalization should feel natural. Sentence case, lowercase, fragments, and inconsistent capitalization are all acceptable if they fit the clip.
- No emotional labels ("hilarious", "wholesome", "satisfying") — show the thing, not the feeling
- No narration ("watch as", "this video shows", "the moment when") — drop the viewer in directly
- No emojis unless one specific emoji is doing real comedic work (most titles don't need one)
- Fragments, slightly wrong grammar, weird specificity — all acceptable, all real

Rank titles by this priority:
1. Specificity
2. Curiosity gap
3. Visual imagery
4. Emotional reaction
5. Brevity

The first criterion dominates all others.
A highly specific title should beat a clever but generic title every time.

Do not rush.
Think of at least 10 possible titles privately.
Keep only the strongest 5.

The title should feel like a sentence that escaped from the middle of a conversation.

Examples:
✅
- the dog won't go near that hallway anymore
- he kept looking behind the vending machine
- nobody reacted when the alarm went off

❌
- funniest dog ever
- wait until the end
- crazy moment caught on camera

When in doubt: mention a strange detail, leave out the explanation, and trust the viewer's curiosity."""

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
                        temperature=1.0,
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

                if best_title:
                    break

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

            _DESCRIPTION_STYLES = [
                "observation (e.g., 'he was doing this for five minutes before i started recording')",
                "mini-story (e.g., 'apparently this happens every morning')",
                "opinion / reaction (e.g., 'i still don't know why the dog hates this hallway')",
                "deadpan statement / extra context (e.g., 'the cashier noticed it instantly' or 'the weird part is nobody else reacted')",
                "question (not generic engagement like 'what do you think', but a specific question related to the scene. Note: Questions should be used less often than other styles — do not default to this style.)",
            ]
            chosen_style = random.choices(
                _DESCRIPTION_STYLES,
                weights=[3, 3, 3, 3, 1]
            )[0]

            phase3_prompt = f"""You are writing the description for ONE YouTube Short.
Write it the way an actual person posting THIS specific video would — casual and specific.

VIDEO ANALYSIS:
- Key moment: {analysis['key_moment']}
- Subjects: {', '.join(analysis['subject_entities'])}

CHOSEN TITLE: "{best_title}"

REQUIRED DESCRIPTION STYLE FOR THIS VIDEO: {chosen_style}

The description should feel like the second thing someone says after showing a friend the clip.
(If the title is the sentence that escaped from the middle of a conversation, the description is the next sentence in that conversation.)

- The description's job is to reward curiosity created by the title. Add exactly one new piece of information (e.g., if Title is "the dog won't go near that hallway anymore", a Good description is "he's avoided that corner for three days now").
- Do not explain everything or solve the mystery. Never restate the title.
- Keep it sounding like it was typed in under 10 seconds. Fragments, sentence fragments, lowercase, and incomplete thoughts are acceptable.
- Never use phrasing like "this clip", "this video", "this one", "caught on camera", or typical AI-intro phrasing ("In this video...", "Welcome back...", "Here's what happens...").
- Never ask generic engagement questions ("what do you think?", "like and subscribe").
- Avoid sounding like corporate SEO copy.
- If it could be copied onto another upload with minimal changes, reject it. It must belong uniquely to THIS video.

Examples of GOOD description lines:
• he was doing this for five minutes before i started recording
• apparently this happens every morning
• i still don't know why the dog hates this hallway
• the cashier noticed it instantly
• the weird part is nobody else reacted

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
5. Did I add exactly one new piece of information? (If no, discard)

HASHTAG RULES (placed at the end on a single line, same field):
- 10 to 13 hashtags total, niche-first then mainstream. Vary count and split slightly each time.
- Priority: (1) exact subjects in the video, (2) exact category of content, (3) broad discovery hashtags.
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

THUMBNAIL: Most dramatic frame, specific timestamp, overlay suggestion for max CTR."""

            metadata = None
            quota_exhausted_models = 0
            for model in _FALLBACK_MODELS:
                try:
                    metadata = await _call_gemini(
                        client, model,
                        contents=["Produce the PublishingPackage."],
                        schema=PublishingPackage,
                        max_tokens=1500,
                        temperature=0.8,
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
                "target_emotion": analysis["emotional_arc"],
                "hook_style": analysis["core_hook"],
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
    result = await generate_metadata_async(
        video_path="videos/meme1.mp4",
        vibe="funny animal meme",
    )
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("Metadata generation returned None — quota exhausted or upload failed. Check logs.")


if __name__ == "__main__":
    asyncio.run(_test())