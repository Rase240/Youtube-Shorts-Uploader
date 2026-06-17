# YouTube Bot — Automated Upload with LLM Metadata

Async pipeline that clips videos with FFmpeg, generates SEO metadata via Gemini 3.5 Flash, and uploads to YouTube.

## Setup

```bash
pip install -r requirements.txt
```

Set environment variables:
```bash
export GEMINI_API_KEY=your_gemini_api_key
```

Place your `client_secrets.json` (Google OAuth) in the project root.

## Usage

Edit the jobs list in `main.py`, then run:

```bash
python main.py
```

OAuth browser window opens on first run — token is saved to `token.pickle` after that.

## File Structure

```
youtube_bot/
├── metadata.py         # Gemini 3.5 Flash metadata generation
├── video_processor.py  # FFmpeg clipping + subtitle burning
├── uploader.py         # YouTube OAuth + resumable upload
├── scheduler.py        # Async batch job runner
├── main.py             # Entry point
└── requirements.txt
```

## Notes

- Videos upload as `private` by default — change in `main.py` after testing
- YouTube free tier: 10,000 quota units/day (~6 uploads)
- `max_concurrent=2` in `run_batch()` is safe for quota limits
