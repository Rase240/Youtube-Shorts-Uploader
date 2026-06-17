# YouTube Bot — Automated Upload with LLM Metadata

An asynchronous Python pipeline that downloads videos from Google Drive, generates SEO-optimized metadata (titles, descriptions, tags) using the Gemini API, and uploads them directly to YouTube.

## Features
- **Google Drive Integration**: Downloads video files directly via shareable links.
- **AI Metadata Generation**: Automatically generates engaging titles, descriptions, and tags tailored to the video's "vibe" and genre using the Gemini API.
- **Async Batch Uploads**: Uploads multiple videos concurrently to YouTube while respecting API rate limits.
- **Resumable Uploads**: Uses chunked resumable uploads for reliability.

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

## Usage

1. Open `main.py` and edit the `jobs` list to add your videos:
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

2. Run the bot:
   ```bash
   python main.py
   ```

**Note on First Run**: A browser window will open asking you to log in to your Google/YouTube account and authorize the application. Once approved, an authentication token is saved to `token.pickle` so you won't need to log in again for future uploads.

## File Structure

```text
youtube_bot/
├── auth.py             # Handles Google OAuth2 flow and token caching
├── drive.py            # Downloads videos from Google Drive links
├── main.py             # Entry point; define your upload jobs here
├── metadata.py         # Interfaces with Gemini API for SEO metadata
├── scheduler.py        # Manages the async job pipeline and concurrency
├── uploader.py         # YouTube API resumable video upload logic
├── requirements.txt    # Python dependencies
├── .env.example        # Template for environment variables
└── .gitignore          # Keeps secrets and temp files out of version control
```

## Important Notes & Quotas

- **Privacy**: Videos upload as `private` by default. You can change this in `main.py` by adding `privacy="public"` to your `Job`.
- **YouTube Quotas**: The free tier of the YouTube Data API provides 10,000 quota units per day. A video upload costs 1,600 units, allowing for about **6 uploads per day**.
- **Concurrency**: `max_concurrent=2` in `main.py` is recommended to stay within safe API request limits.
