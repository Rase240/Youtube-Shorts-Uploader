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
    """Raised when Gemini quota is exhausted after all retries.

    Unlike transient UNAVAILABLE errors, switching models won't help here —
    both use the same API key and the same account-level quota. The caller
    should queue the video for later rather than crashing or hammering the API.
    """
    pass


# ── Pydantic Schemas ─────────────────────────────────────────────────────────

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
    title_setups: list[str] = Field(
        ...,
        description=(
            "Generate exactly 3 raw title-seed observations — the most specific, concrete things "
            "you noticed that a title writer can build from. Each should be a brief, visual observation "
            "(NOT a polished title), e.g.: 'the cat slowly turns its head right before the noise hits' "
            "or 'the person's expression goes from confident to pure panic in under one second'. "
            "Focus on the single most shareable moment and what makes it funny/shocking/relatable. "
            "Be hyper-specific and visual — the more granular, the more useful for the title writer."
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
    def cap_title(cls, v: str) -> str:
        if len(v) > 55:
            truncated = v[:52]
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
    def cap_tags(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip("#").strip() for t in v]
        if len(cleaned) > 15:
            logger.warning(
                f"[PHASE 3] Model returned {len(cleaned)} tags, truncating to 15. "
                f"Dropped: {cleaned[15:]}"
            )
            return cleaned[:15]
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
    if len(words) >= 5:
        capitalized = sum(1 for w in words[1:] if len(w) > 2 and w[0].isupper())
        if capitalized / (len(words) - 1) > 0.75:
            return "Title looks too formally capitalized (not Gen-Z voice)"

    return None


def _log_hashtag_and_tag_counts(
    description: str,
    tags: list[str],
    niche_hashtag_count: int,
    niche_tag_count: int,
) -> None:
    """Logs hashtag/tag counts and verifies niche/mainstream split. Non-fatal."""
    hashtags = re.findall(r"#\w+", description)
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
            if response.parsed:
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
                    f"Daily limit likely hit — skipping this video."
                )
                raise QuotaExhaustedError(f"Gemini quota exhausted after {max_attempts} retries: {e}") from e

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

    logger.info(f"[GEMINI] Uploading video: {video_path}")
    try:
        video_file = await client.aio.files.upload(file=video_path)
        logger.info("[GEMINI] Uploaded. Waiting for processing...")

        try:
            max_polls = 60
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
- 3 raw title-seed observations — the most specific, granular things you noticed
  that could be the kernel of a great title. NOT polished titles — raw observations.
  Think: "what's the one-second thing you'd describe to a friend to explain why you sent them this?"

Be specific and visual. Reference exact moments in the video."""

            analysis = None
            for model in ["gemini-3.5-flash", "gemini-3.1-flash-lite"]:
                try:
                    analysis = await _call_gemini(
                        client, model,
                        contents=[video_file],
                        schema=VideoAnalysis,
                        max_tokens=1500,
                        temperature=0.7,
                        sys_instruct=phase1_prompt,
                    )
                    if analysis:
                        logger.info(f"[PHASE 1] Done. Core hook: {analysis.get('core_hook', 'N/A')}")
                        logger.info(f"[PHASE 1] Title seeds: {analysis.get('title_setups', [])}")
                        break
                except QuotaExhaustedError:
                    raise  # account-level — no point trying the next model
                except Exception as e:
                    logger.warning(f"[PHASE 1] {model} failed: {e}")
                    continue

            if not analysis:
                raise RuntimeError("Phase 1 (video analysis) failed on all models.")

            # ════════════════════ PHASE 2: TITLE GENERATION ════════════════════
            logger.info("[PHASE 2] Generating title candidates...")

            # Techniques that reliably produce grounded, specific titles.
            # Removed "incomplete comparison" (produces fragment garbage like "funnier than—")
            # and "deadpan label" (produces corporate-speak like "individual disrupts equilibrium").
            # Both fail because they're abstract methods that don't anchor to specific content.
            _ALL_TECHNIQUES = [
                "curiosity gap: name the exact setup visible in the video, withhold the outcome entirely — the viewer must watch to find out what happens",
                "accusation/callout: address someone actually visible in the video as if catching them in the act, naming their specific action",
                "understatement: describe the most extreme visible moment using the flattest, most casual language possible — make big feel small",
                "reaction fragment: write the exact first thing you'd text a friend right after watching this specific clip, grounded in what actually happens on screen",
                "specific detail anchor: lead with one hyper-specific visual detail from the video that's weird or funny out of context — nothing else, no explanation",
                "false confidence: make a bold, specific claim about someone or something actually visible in this video that a viewer will immediately want to verify or argue with",
                "second-person drop-in: put the viewer directly into the exact situation shown — use 'you' for the specific thing that happens in this video",
                "scene contrast: name two things happening simultaneously in this video that shouldn't go together — leave it unresolved, let the absurdity do the work",
                "peak expression: describe someone's face/body language/reaction at the exact peak moment in the most specific visual terms possible",
            ]
            chosen_techniques = random.sample(_ALL_TECHNIQUES, 5)
            techniques_block = "\n".join(f"- {t}" for t in chosen_techniques)
            seeds_block = "\n".join(f"  • {s}" for s in analysis.get("title_setups", []))

            phase2_prompt = f"""You are writing ONE YouTube Shorts title. Watch this video carefully.
Everything in this title must come from what's actually visible in THIS specific video.
There is no house style to fall back on — every word has to be earned by the content.

VIDEO ANALYSIS:
KEY MOMENT: {analysis['key_moment']}
EMOTIONAL ARC: {analysis['emotional_arc']}
SHAREABILITY: {analysis['shareability_factor']}
CORE HOOK: {analysis['core_hook']}
SUBJECTS: {', '.join(analysis['subject_entities'])}

RAW TITLE SEEDS (specific observations from the analyst — use these as your starting material):
{seeds_block}

Apply each technique below to the SPECIFIC content in this video. Ground every title in a real
moment or detail that's actually visible. Do not write a generic version of the technique:

{techniques_block}

THE ONLY RULE THAT MATTERS: a title must CREATE a gap, not CLOSE one.
If a viewer reads it and already knows how the video ends — you've failed.
State the SETUP only. The outcome, punchline, and twist must not appear in the title.

ALL OTHER RULES (every candidate must satisfy all of these):
- Under 55 characters — hard limit, mobile truncates here
- Lowercase throughout — sounds like something typed in 4 seconds, not composed
- No emotional labels ("hilarious", "wholesome", "satisfying") — show the thing, not the feeling
- No narration ("watch as", "this video shows", "the moment when") — drop the viewer in directly
- No emojis unless one specific emoji is doing real comedic work (most titles don't need one)
- Fragments, slightly wrong grammar, weird specificity — all acceptable, all real

After writing all 5, rank by which creates the STRONGEST unresolved question.
The best title is usually the most specific and the most incomplete."""

            best_title = None
            last_title_data = None  # kept for quality-gate fallback

            for model in ["gemini-3.5-flash", "gemini-3.1-flash-lite"]:
                for attempt in range(3):
                    try:
                        title_data = await _call_gemini(
                            client, model,
                            contents=[video_file],
                            schema=TitleCandidates,
                            max_tokens=2000,
                            temperature=1.0,
                            sys_instruct=phase2_prompt,
                            max_attempts=3,
                        )
                        if title_data:
                            last_title_data = title_data
                            candidate = title_data.get("best_title", "")
                            quality_issue = _check_title_quality(candidate)

                            if quality_issue:
                                logger.warning(
                                    f"[PHASE 2] best_title rejected ({model}, attempt {attempt + 1}/3): "
                                    f"{quality_issue} — '{candidate}'"
                                )
                                # Scan alternates before re-prompting
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

                    except QuotaExhaustedError:
                        raise
                    except Exception as e:
                        logger.warning(f"[PHASE 2] {model} completely failed: {e}")
                        break  # _call_gemini exhausted its retries — try next model

                if best_title:
                    break

            # Quality-gate fallback: a slightly imperfect title is always better than
            # crashing the pipeline and skipping the upload entirely. Log it loudly so
            # you can tune the gate or the prompts later.
            if not best_title and last_title_data:
                fallback = last_title_data.get("best_title", "")
                if fallback:
                    logger.warning(
                        f"[PHASE 2] All quality checks failed — using raw best_title as fallback: '{fallback}'. "
                        f"Check logs to tune quality gate or prompts."
                    )
                    best_title = fallback

            if not best_title:
                raise RuntimeError("Phase 2 (title generation) failed to produce any title.")

            # ════════════════════ PHASE 3: SUPPORTING METADATA ════════════════════
            logger.info("[PHASE 3] Generating description, tags, and engagement metadata...")

            _DESCRIPTION_SHAPES = [
                (
                    "ONE LINE ONLY before the hashtags: a single punchy line that's either "
                    "a callout, a question, or a flat statement of the absurd thing that "
                    "happened. No build-up, no context paragraph — just the line, then hashtags."
                ),
                (
                    "TWO LINES before the hashtags: line 1 is a hook or reaction, line 2 adds "
                    "one specific, slightly tangential detail (almost an aside) that naturally "
                    "includes a keyword — not a recap of the video, more like something you'd "
                    "add as an afterthought."
                ),
                (
                    "QUESTION-LED: open by asking the viewer something directly related to the "
                    "key moment (not 'what do you think' — something specific they'd actually "
                    "want to answer), then one short follow-up line, then hashtags."
                ),
            ]
            chosen_shape = random.choice(_DESCRIPTION_SHAPES)

            phase3_prompt = f"""You are writing the description for ONE YouTube Short.
Write it the way an actual person posting THIS specific video would — not a template.

VIDEO ANALYSIS:
- Key moment: {analysis['key_moment']}
- Core hook: {analysis['core_hook']}
- Subjects: {', '.join(analysis['subject_entities'])}

CHOSEN TITLE: "{best_title}"

DESCRIPTION SHAPE FOR THIS ONE: {chosen_shape}

Whatever shape you use, the text before the hashtags must:
- Never restate the title — add NEW information or a different angle on it
- Never use AI-intro phrasing ("In this video...", "Welcome back...", "Here's what happens...")
- Never explain that something is funny/wholesome/satisfying — describe the specific thing, let it land
- Sound like it was typed in 10 seconds on a phone, not composed

HASHTAG LINE (after the text, same field):
- 10 to 13 hashtags total, niche-first then mainstream. Vary count and split slightly each time.
  Roughly 55-65% NICHE (tied to this video's specific subjects/characters/objects),
  remainder MAINSTREAM for broad discovery (real high-view videos in this category use these).
- This is a NEW account — mainstream hashtags are required for discovery reach.

TAG RULES ("tags" field, separate from hashtags):
- 12 to 15 tags, niche-first then mainstream. Vary count and split slightly each time.
  Roughly 55-65% NICHE — specific multi-word phrases for: {', '.join(analysis['subject_entities'])}
  Remainder MAINSTREAM — broader category/discovery tags real viral videos in this niche rank for.
- Ask: "what would someone type to find THIS EXACT video?" (niche) vs
  "what would someone type to find content LIKE this?" (mainstream)

PINNED COMMENT: Short opinionated question that FORCES replies. Under 15 words.
Sound like a nosy person stirring something, not a survey.

THUMBNAIL: Most dramatic frame, specific timestamp, overlay suggestion for max CTR."""

            metadata = None
            for model in ["gemini-3.5-flash", "gemini-3.1-flash-lite"]:
                try:
                    metadata = await _call_gemini(
                        client, model,
                        contents=["Generate Phase 3 metadata based on the system instructions."],
                        schema=SupportingMetadata,
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
                    raise
                except Exception as e:
                    logger.warning(f"[PHASE 3] {model} failed: {e}")
                    continue

            if not metadata:
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
        logger.error(
            f"[QUOTA] Gemini quota exhausted — skipping '{video_path}'. "
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