#!/usr/bin/env python3
# Copyright (C) 2026 swatah.ai. All rights reserved.
#
# This software is dual-licensed:
# 1. GNU General Public License v3.0 (GPLv3)
# 2. A proprietary license for commercial use.
#
# You may use this software under the terms of the GPLv3 if you are using it
# for non-commercial purposes. For commercial usage, a separate commercial 
# license must be obtained from swatah.ai (info@swatah.ai).
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
# for more details.
#
# Trademarks: All trademarks, service marks, and logos are the property of 
# their respective owners.

"""
YouTube video downloader script using yt-dlp.
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path

def install_ytdlp():
    """Install yt-dlp using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
        print("yt-dlp installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install yt-dlp: {e}")
        return False

def check_and_install_ytdlp():
    """Check if yt-dlp is available, ask to install if not."""
    try:
        import yt_dlp
        return True
    except ImportError:
        print("yt-dlp is not installed.")
        response = input("Do you want to install yt-dlp now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            print("Installing yt-dlp...")
            return install_ytdlp()
        else:
            print("Error: yt-dlp is not installed. Please install it using:")
            print("pip install yt-dlp")
            return False

def download_youtube_video(url, output_dir=None, format_option=None):
    """
    Download a YouTube video using yt-dlp.

    Args:
        url (str): YouTube video URL
        output_dir (str): Directory to save the video (default: current directory)
        format_option (str): Format specification for yt-dlp (default: best quality)
    """
    # Set default output directory to current directory if not specified
    if output_dir is None:
        output_dir = os.getcwd()

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Configure yt-dlp options
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
    }

    # Add format option if specified
    if format_option:
        ydl_opts['format'] = format_option
    else:
        # Default to best quality
        ydl_opts['format'] = 'best'

    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading video from: {url}")
            ydl.download([url])
            print("Download completed successfully!")
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        sys.exit(1)

def main():
    # Check and install yt-dlp if needed
    if not check_and_install_ytdlp():
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Download YouTube videos using yt-dlp')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('-o', '--output', help='Output directory (default: current directory)')
    parser.add_argument('-f', '--format', help='Video format (e.g., "best", "worst", "720p")')
    parser.add_argument('--list-formats', action='store_true',
                       help='List available formats for the video and exit')

    args = parser.parse_args()

    if args.list_formats:
        try:
            import yt_dlp
            ydl_opts = {'listformats': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([args.url])
        except Exception as e:
            print(f"Error listing formats: {str(e)}")
            sys.exit(1)
    else:
        download_youtube_video(args.url, args.output, args.format)

if __name__ == '__main__':
    main()