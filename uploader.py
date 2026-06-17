import asyncio
import logging
import os
import pickle
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_PROJECT_DIR, ".env"))

from auth import get_credentials

DEFAULT_PRIVACY = os.getenv("DEFAULT_PRIVACY", "public")

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# Genre → YouTube Category ID
GENRE_CATEGORY_MAP = {
    "gaming":        "20",
    "music":         "10",
    "education":     "27",
    "tech":          "28",
    "entertainment": "24",
    "travel":        "19",
    "sports":        "17",
    "news":          "25",
    "comedy":        "23",
    "default":       "22",   # People & Blogs
}


class UploadConfig(BaseModel):
    video_path: str
    title: str
    description: str
    tags: list[str]
    genre: str = "default"
    default_privacy: str = DEFAULT_PRIVACY
    made_for_kids: bool = False

    @property
    def category_id(self) -> str:
        key = self.genre.lower().split("&")[0].strip()
        return GENRE_CATEGORY_MAP.get(key, GENRE_CATEGORY_MAP["default"])


def get_youtube_client():
    creds = get_credentials()
    return build("youtube", "v3", credentials=creds)


async def upload_video(config: UploadConfig) -> Optional[str]:
    """
    Runs blocking YouTube upload in executor to keep async loop free.
    Returns video ID on success, None on failure.
    """
    def _blocking_upload():
        youtube = get_youtube_client()

        body = {
            "snippet": {
                "title": config.title,
                "description": config.description,
                "tags": config.tags,
                "categoryId": config.category_id,
            },
            "status": {
                "privacyStatus": config.default_privacy,
                "madeForKids": config.made_for_kids,
            }
        }

        media = MediaFileUpload(
            config.video_path,
            mimetype="video/mp4",
            chunksize=256 * 1024,   # 256KB chunks
            resumable=True
        )

        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )

        # Resumable upload with progress
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                logger.info(f"Upload progress: {int(status.progress() * 100)}%")

        return response.get("id")

    try:
        loop = asyncio.get_event_loop()
        video_id = await loop.run_in_executor(None, _blocking_upload)
        logger.info(f"Upload complete. Video ID: {video_id}")
        return video_id
    except HttpError as e:
        logger.error(f"YouTube API error: {e.resp.status} - {e.content}")
        return None
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return None
