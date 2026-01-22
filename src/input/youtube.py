"""YouTube audio downloader utilities."""

import tempfile
import re
from pathlib import Path
from typing import Optional
import yt_dlp


class YouTubeDownloader:
    """Handles downloading audio from YouTube URLs."""

    # Regex pattern for YouTube URLs
    YOUTUBE_PATTERN = re.compile(
        r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    )

    def __init__(self, output_format: str = "wav"):
        """
        Initialize YouTubeDownloader.

        Args:
            output_format: Audio format to download (wav, mp3, etc.)
        """
        self.output_format = output_format

    @staticmethod
    def is_youtube_url(path: str) -> bool:
        """
        Check if a string is a YouTube URL.

        Args:
            path: String to check

        Returns:
            True if it's a YouTube URL
        """
        return bool(YouTubeDownloader.YOUTUBE_PATTERN.search(path))

    def download(self, url: str, output_path: Optional[str] = None) -> str:
        """
        Download audio from YouTube URL.

        Args:
            url: YouTube URL
            output_path: Optional path to save file. If None, creates temp file.

        Returns:
            Path to downloaded audio file

        Raises:
            ValueError: If URL is invalid
            RuntimeError: If download fails
        """
        if not self.is_youtube_url(url):
            raise ValueError(f"Invalid YouTube URL: {url}")

        # Create output path if not provided
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = str(Path(temp_dir) / f"youtube_audio.{self.output_format}")

        # Configure yt-dlp options
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_path.replace(f".{self.output_format}", ""),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": self.output_format,
                    "preferredquality": "192",
                }
            ],
            "quiet": False,
            "no_warnings": False,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Return the actual output path with extension
            final_path = output_path if output_path.endswith(
                f".{self.output_format}"
            ) else f"{output_path}.{self.output_format}"

            if not Path(final_path).exists():
                raise RuntimeError(f"Download succeeded but file not found: {final_path}")

            return final_path

        except Exception as e:
            raise RuntimeError(f"Failed to download YouTube audio: {e}")
