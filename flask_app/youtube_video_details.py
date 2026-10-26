"""
Fetch YouTube video and channel details with YouTube Data API v3.

Install:
    pip install google-api-python-client python-dotenv isodate pandas

Create .env:
    YOUTUBE_API_KEY=your_api_key

Run:
    python youtube_video_details.py "https://www.youtube.com/watch?v=VIDEO_ID"
    python youtube_video_details.py VIDEO_ID_1 VIDEO_ID_2
    python youtube_video_details.py VIDEO_ID --output youtube_video_details.csv
"""

import argparse
import os
import re
from pathlib import Path
from typing import Any

import isodate
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def extract_video_id(value: str) -> str:
    """Accept a raw YouTube video ID or a standard video URL."""
    value = value.strip()
    patterns = [
        r"(?:youtube\.com/watch\?.*?\bv=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    raise ValueError(f"Cannot extract a YouTube video ID from {value!r}.")


def get_category_names(youtube: Any, region_code: str = "IN") -> dict[str, str]:
    response = (
        youtube.videoCategories().list(part="snippet", regionCode=region_code).execute()
    )
    return {item["id"]: item["snippet"]["title"] for item in response.get("items", [])}


def fetch_video_details(
    youtube: Any, video_id: str, category_names: dict[str, str]
) -> dict[str, Any]:
    response = (
        youtube.videos()
        .list(part="snippet,contentDetails,statistics", id=video_id)
        .execute()
    )
    items = response.get("items", [])
    if not items:
        raise ValueError(
            f"No accessible video found for {video_id!r}; check the ID and privacy."
        )

    video = items[0]
    snippet = video.get("snippet", {})
    content = video.get("contentDetails", {})
    stats = video.get("statistics", {})
    channel_id = snippet.get("channelId", "")
    channel_title = snippet.get("channelTitle", "")
    subscriber_count = channel_view_count = channel_video_count = None

    if channel_id:
        channel_response = (
            youtube.channels().list(part="snippet,statistics", id=channel_id).execute()
        )
        channel_items = channel_response.get("items", [])
        if channel_items:
            channel = channel_items[0]
            channel_stats = channel.get("statistics", {})
            # Subscriber count may be hidden by the channel owner.
            subscriber_count = (
                int(channel_stats["subscriberCount"])
                if "subscriberCount" in channel_stats
                else None
            )
            channel_view_count = (
                int(channel_stats["viewCount"])
                if "viewCount" in channel_stats
                else None
            )
            channel_video_count = (
                int(channel_stats["videoCount"])
                if "videoCount" in channel_stats
                else None
            )
            channel_title = channel.get("snippet", {}).get("title", channel_title)

    category_id = snippet.get("categoryId")
    duration_iso = content.get("duration")
    duration_seconds = (
        int(isodate.parse_duration(duration_iso).total_seconds())
        if duration_iso
        else None
    )

    return {
        "video_id": video.get("id"),
        "video_url": f"https://www.youtube.com/watch?v={video.get('id')}",
        "title": snippet.get("title"),
        "description": snippet.get("description"),
        "tags": ", ".join(snippet.get("tags", [])),
        "category": category_names.get(category_id, category_id),
        "category_id": category_id,
        "published_at": snippet.get("publishedAt"),
        "channel_id": channel_id,
        "channel_title": channel_title,
        "subscriber_count": subscriber_count,
        "channel_view_count": channel_view_count,
        "channel_video_count": channel_video_count,
        "duration_iso8601": duration_iso,
        "duration_seconds": duration_seconds,
        "view_count": int(stats["viewCount"]) if "viewCount" in stats else None,
        "like_count": int(stats["likeCount"]) if "likeCount" in stats else None,
        "comment_count": (
            int(stats["commentCount"]) if "commentCount" in stats else None
        ),
    }


def main() -> None:
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "YOUTUBE_API_KEY is missing. Add it to .env or your environment."
        )

    parser = argparse.ArgumentParser(
        description="Fetch YouTube video and channel statistics."
    )
    parser.add_argument("videos", nargs="+", help="Video IDs or YouTube video URLs.")
    parser.add_argument(
        "--output", default="youtube_video_details.csv", help="CSV output path."
    )
    parser.add_argument(
        "--region", default="IN", help="Category-name region code (default: IN)."
    )
    args = parser.parse_args()

    youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
    categories = get_category_names(youtube, args.region)
    rows = []
    for value in args.videos:
        try:
            video_id = extract_video_id(value)
            rows.append(fetch_video_details(youtube, video_id, categories))
            print(f"Fetched video: {video_id}")
        except (ValueError, HttpError) as exc:
            print(f"Skipping {value!r}: {exc}")

    if not rows:
        raise RuntimeError("No video details were retrieved.")

    dataframe = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print("\nVideo details:")
    print(dataframe.to_string(index=False))
    print(f"\nSaved {len(rows)} row(s) to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
