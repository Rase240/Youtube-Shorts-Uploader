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
    parent = os.path.dirname(abs_output)
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        with requests.Session() as session:
            logger.info(f"[DRIVE] Downloading {file_id}...")

            # First request — may get a virus scan warning page for large files
            response = session.get(download_url, stream=True, timeout=30)
            response.raise_for_status()

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
                    response = session.get(download_url, stream=True, timeout=30)
                    response.raise_for_status()
                    
                    # Double-check that we didn't receive another HTML page after confirmation
                    if "text/html" in response.headers.get("Content-Type", ""):
                        logger.error("[DRIVE] Still got HTML warning page after sending confirm token. File might be restricted or require sign-in.")
                        return False
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
        if os.path.exists(abs_output):
            try:
                os.remove(abs_output)
            except Exception as cleanup_err:
                logger.warning(f"[DRIVE] Failed to clean up partial file {abs_output}: {cleanup_err}")
        return False


async def download_video(drive_url: str, output_path: str) -> bool:
    """Downloads a public Google Drive video. Runs in a thread executor."""
    return await asyncio.to_thread(_blocking_download, drive_url, output_path)


def _blocking_download_discord(url: str, output_path: str, max_size_bytes: int) -> bool:
    """Downloads a file from a Discord attachment or direct URL with size protection."""
    abs_output = os.path.join(_PROJECT_DIR, output_path) if not os.path.isabs(output_path) else output_path
    parent = os.path.dirname(abs_output)
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        logger.info(f"[DISCORD] Downloading from URL...")
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()

        # Check content-length header if present
        content_length = r.headers.get("Content-Length")
        if content_length and int(content_length) > max_size_bytes:
            logger.error(f"[DISCORD] File exceeds size limit: {content_length} bytes (limit is {max_size_bytes} bytes)")
            return False

        # Validate content type
        content_type = r.headers.get("Content-Type", "")
        if content_type and not content_type.startswith("video/"):
            logger.warning(f"[DISCORD] Unexpected content type: {content_type}")

        total = 0
        exceeded = False
        with open(abs_output, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:
                    total += len(chunk)
                    if total > max_size_bytes:
                        logger.error(f"[DISCORD] Download exceeded max size limit of {max_size_bytes} bytes. Aborting.")
                        exceeded = True
                        break
                    f.write(chunk)

        if exceeded:
            if os.path.exists(abs_output):
                os.remove(abs_output)
            return False

        size_mb = total / (1024 * 1024)
        logger.info(f"[DISCORD] Download complete. {size_mb:.1f} MB saved to {abs_output}")
        return True

    except Exception as e:
        logger.error(f"[DISCORD] Download failed: {e}")
        if os.path.exists(abs_output):
            try:
                os.remove(abs_output)
            except Exception as cleanup_err:
                logger.warning(f"[DISCORD] Failed to clean up partial file {abs_output}: {cleanup_err}")
        return False


async def download_discord_video(url: str, output_path: str, max_size_bytes: int = 2 * 1024 * 1024 * 1024) -> bool:
    """Downloads a Discord video. Runs in a thread executor."""
    return await asyncio.to_thread(_blocking_download_discord, url, output_path, max_size_bytes)
