import os
import logging
from PIL import Image, ImageFilter
import PIL.Image

# Compatibility shim: Pillow 10+ removed ANTIALIAS, but moviepy 1.x still uses it internally.
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from moviepy.editor import VideoFileClip, CompositeVideoClip
import numpy as np

logger = logging.getLogger(__name__)

def get_video_info(video_path: str) -> dict:
    """Returns duration, width, height, and boolean if horizontal."""
    try:
        with VideoFileClip(video_path) as clip:
            info = {
                "duration": clip.duration,
                "width": clip.w,
                "height": clip.h,
                "is_horizontal": clip.w > clip.h
            }
            return info
    except Exception as e:
        logger.error(f"Failed to get video info for {video_path}: {e}")
        return {"duration": 0, "width": 0, "height": 0, "is_horizontal": False}


def _blur_image(image_array, radius=15, darken_factor=0.6):
    """Blurs and darkens an image array using Pillow."""
    pil_im = Image.fromarray(image_array)
    pil_im = pil_im.filter(ImageFilter.GaussianBlur(radius=radius))
    # Darken for better contrast with foreground
    pil_im = pil_im.point(lambda p: p * darken_factor)
    return np.array(pil_im)


def pad_video_for_shorts(video_path: str) -> str:
    """
    Pads a horizontal video to 9:16 vertical format using a blurred background.
    Returns the path to the newly processed video.
    """
    output_path = video_path.rsplit('.', 1)[0] + "_padded.mp4"
    logger.info(f"Padding horizontal video {video_path} to vertical Shorts format...")
    
    try:
        clip = VideoFileClip(video_path)
        target_w, target_h = 1080, 1920
        
        # 1. Foreground (original video resized to fit width 1080)
        fg_clip = clip.resize(width=target_w)
        
        # 2. Background (resize to fill 1920 height, then crop width to 1080)
        bg_clip = clip.resize(height=target_h)
        x_center = bg_clip.w / 2
        bg_clip = bg_clip.crop(x1=x_center - target_w/2, width=target_w)
        
        # Apply blur to each frame
        bg_clip = bg_clip.fl_image(lambda frame: _blur_image(frame))
        
        # 3. Composite them together
        final_clip = CompositeVideoClip([bg_clip, fg_clip.set_position("center")])
        
        # Write file with aac audio to ensure compatibility
        final_clip.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac",
            logger=None # Suppress moviepy terminal spam
        )
        
        clip.close()
        fg_clip.close()
        bg_clip.close()
        final_clip.close()
        
        logger.info(f"Successfully processed video into {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to pad video: {e}")
        # If processing fails, fallback to returning original path
        return video_path
