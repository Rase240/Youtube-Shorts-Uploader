import asyncio
import logging
import os
import re
import requests

logger = logging.getLogger(__name__)

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_file_id(url: str) -> str:
    """Extracts the Google Drive file ID from a shareable link."""
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return url


def _blocking_download(drive_url: str, output_path: str) -> bool:
    """Downloads a public Google Drive file via direct download link."""
    file_id = extract_file_id(drive_url)
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    abs_output = os.path.join(_PROJECT_DIR, output_path) if not os.path.isabs(output_path) else output_path
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    try:
        session = requests.Session()
        logger.info(f"[DRIVE] Downloading {file_id}...")

        # First request — may get a virus scan warning page for large files
        response = session.get(download_url, stream=True)

        # Check for the "confirm download" token (large file warning)
        if "text/html" in response.headers.get("Content-Type", ""):
            # Google asks for confirmation on large files
            confirm_token = None
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    confirm_token = value
                    break

            if confirm_token:
                download_url = f"{download_url}&confirm={confirm_token}"
                response = session.get(download_url, stream=True)
            else:
                logger.error("[DRIVE] Received HTML page but no download_warning cookie found. File might be restricted or require sign-in.")
                return False

        # Stream the file to disk
        total = 0
        with open(abs_output, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    total += len(chunk)

        size_mb = total / (1024 * 1024)
        logger.info(f"[DRIVE] Download complete. {size_mb:.1f} MB saved to {abs_output}")
        return True

    except Exception as e:
        logger.error(f"[DRIVE] Download failed: {e}")
        return False


async def download_video(drive_url: str, output_path: str) -> bool:
    """Downloads a public Google Drive video. Runs in a thread executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _blocking_download, drive_url, output_path)
