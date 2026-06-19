# YouTube Bot — Automated Upload with LLM Metadata

An asynchronous Python pipeline that downloads videos from Google Drive/Discord attachments, generates SEO-optimized metadata (titles, descriptions, tags) using the Gemini API, and uploads them directly to YouTube.

## Features
- **Google Drive & Discord Support**: Downloads video files directly via shareable drive links or direct discord URLs.
- **AI Metadata Generation**: Automatically generates engaging titles, descriptions, and tags tailored to the video's "vibe" and genre using the Gemini API.
- **Auto-Retry on Malformed JSON**: If Gemini outputs malformed JSON, the script automatically logs the raw text for debugging and retries generating metadata up to 3 times per model.
- **Robust Rotating File Logs**: Logs all pipeline operations to both the console and a local `youtube_bot.log` file, which is self-managing (5MB size cap, 3 backups max) to prevent disk bloat.
- **Leak-Proof Cleanup**: Uses strict `try...finally` blocks to guarantee that all temporary download videos are deleted, preventing disk clutter even if an upload crashes.
- **Independent CLI Tooling**: Includes a feature-rich CLI for single-video uploads, listing, privacy updates, and deletions standalone from your console.
- **Async Batch Uploads**: Uploads multiple videos concurrently to YouTube while respecting API rate limits.

---

## Setup & Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Copy `.env.example` to `.env` and fill in your Gemini API key:
   ```bash
   cp .env.example .env
   ```
   *Edit `.env` and set `GEMINI_API_KEY=your_gemini_api_key_here`*

3. **Google API Credentials**
   - Go to the [Google Cloud Console](https://console.cloud.google.com/).
   - Create a new project and enable the **YouTube Data API v3**.
   - Create **OAuth 2.0 Client ID** credentials (choose "Desktop App").
   - Download the JSON file, rename it to `client_secrets.json`, and place it in the project root folder.

---

## Standalone Usage

You do not need a Discord bot to use this tool! It runs fully standalone.

### 1. Batch Uploads (`main.py`)
Open `main.py` and edit the `jobs` list to add your videos:
```python
jobs = [
    Job(
        drive_url="https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing", 
        vibe="your video vibe/niche",
        genre="comedy",
    ),
]
```
*(You can also use local files by specifying `video_path="videos/your_video.mp4"` instead of `drive_url`)*

Run the batch runner:
```bash
python main.py
```

### 2. Single Operations & CLI (`youtube_cli.py`)
Run standalone operations from your command line:
* **Upload**:
  ```bash
  python youtube_cli.py upload --vibe "comedy meme" --drive_url "GDRIVE_URL" --privacy public
  ```
* **List Recent Uploads (with view, like, comment stats)**:
  ```bash
  python youtube_cli.py list
  ```
* **Change Video Privacy**:
  ```bash
  python youtube_cli.py setprivacy --video_id "VIDEO_ID" --privacy private
  ```
* **Delete Video**:
  ```bash
  python youtube_cli.py delete --video_id "VIDEO_ID"
  ```

---

## Discord Bot Integration

When integrated with the companion Discord bot:
- **`yt upload [attachment/url] vibe: <niche>`**: Uploads Drive links or attachments directly to YouTube shorts.
- **`yt list`**: Shows an interactive paginated list of your 50 most recent uploads and their performance stats.
- **`yt setprivacy <id/url> <privacy>`**: Updates privacy settings from Discord.
- **`yt delete <id/url>`**: Deletes a video.
- **`yt logs`**: Previews the last 20 lines of the upload logs directly in chat.
- **`yt logs --download`**: Sends the full `youtube_bot.log` file to you as an attachment.

---

## File Structure

```text
youtube_bot/
├── auth.py             # Handles Google OAuth2 flow and token caching
├── drive.py            # Downloads videos from Google Drive links
├── main.py             # Entry point; define your upload jobs here
├── metadata.py         # Interfaces with Gemini API for SEO metadata and sets up logs
├── scheduler.py        # Manages the async job pipeline and concurrency
├── uploader.py         # YouTube API resumable video upload logic
├── youtube_cli.py      # Independent CLI utility for standalone actions
├── youtube_bot.log     # Self-managing rotating log output file
├── requirements.txt    # Python dependencies
├── .env.example        # Template for environment variables
└── .gitignore          # Keeps secrets and logs out of git history
```

---

## Important Notes & Quotas

- **YouTube Quotas**: The free tier of the YouTube Data API provides 10,000 quota units per day. A video upload costs 1,600 units, allowing for about **6 uploads per day**.
- **Log Size limits**: Log file sizes are capped at 5MB, maintaining up to 3 backup versions on a rolling basis. Older logs are purged automatically.
