import os
import json
import logging
import subprocess

logger = logging.getLogger(__name__)


def get_video_info(video_path: str) -> dict:
    """Returns duration, width, height, and boolean if horizontal using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-show_format",
                video_path
            ],
            capture_output=True, text=True, timeout=30
        )
        data = json.loads(result.stdout)

        # Find the video stream
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            logger.error(f"No video stream found in {video_path}")
            return {"duration": 0, "width": 0, "height": 0, "is_horizontal": False}

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        duration = float(data.get("format", {}).get("duration", 0))

        return {
            "duration": duration,
            "width": width,
            "height": height,
            "is_horizontal": width > height
        }
    except Exception as e:
        logger.error(f"Failed to get video info for {video_path}: {e}")
        return {"duration": 0, "width": 0, "height": 0, "is_horizontal": False}


def pad_video_for_shorts(video_path: str) -> str:
    """
    Pads a horizontal video to 9:16 vertical format using a blurred background.
    Uses ffmpeg directly (streams frames, no RAM bloat).
    Returns the path to the newly processed video.
    """
    output_path = video_path.rsplit('.', 1)[0] + "_padded.mp4"
    logger.info(f"Padding horizontal video {video_path} to vertical Shorts format...")

    try:
        # ffmpeg filter:
        #   [bg] = scale up to fill 1080x1920, crop to exact size, blur + darken
        #   [fg] = scale down to fit 1080 width, keep aspect ratio (height auto)
        #   overlay [fg] centered on [bg]
        filter_complex = (
            "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            "boxblur=20:5,"
            "colorlevels=rimax=0.6:gimax=0.6:bimax=0.6[bg];"
            "[0:v]scale=1080:-2:force_original_aspect_ratio=decrease[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-filter_complex", filter_complex,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path
        ]

        logger.info(f"Running ffmpeg command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            logger.error(f"ffmpeg padding failed (exit code {result.returncode}): {result.stderr[-2000:]}")
            return video_path

        logger.info(f"Successfully processed video into {output_path}")
        return output_path

    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg padding timed out for {video_path}")
        return video_path
    except Exception as e:
        logger.error(f"Failed to pad video: {e}")
        return video_path
