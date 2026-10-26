import argparse
import os
import re
from urllib.parse import urlparse, parse_qs

import isodate
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def normalise_url(value):
    value = value.strip()
    return value if value.startswith(("http://", "https://")) else "https://" + value


def extract_video_id(url):
    parsed = urlparse(normalise_url(url))
    host = parsed.netloc.lower()
    if host.endswith("youtu.be"):
        candidate = parsed.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        else:
            match = re.match(r"^/(?:shorts|embed|live)/([A-Za-z0-9_-]{11})", parsed.path)
            candidate = match.group(1) if match else None
    else:
        candidate = None
    return candidate if candidate and re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate) else None


def get_channel(youtube, url_or_id):
    value = url_or_id.strip()
    if value.startswith("UC") and len(value) >= 24 and "/" not in value:
        response = youtube.channels().list(part="snippet,statistics,contentDetails", id=value).execute()
    else:
        parsed = urlparse(normalise_url(value))
        if not parsed.netloc.lower().endswith("youtube.com"):
            raise ValueError("Enter a YouTube channel URL or video URL.")
        parts = [p for p in parsed.path.split("/") if p]
        if not parts:
            raise ValueError("The URL does not contain a channel handle or ID.")
        if parts[0].startswith("@"):
            response = youtube.channels().list(part="snippet,statistics,contentDetails", forHandle=parts[0]).execute()
        elif parts[0] == "channel" and len(parts) > 1:
            response = youtube.channels().list(part="snippet,statistics,contentDetails", id=parts[1]).execute()
        elif parts[0] == "user" and len(parts) > 1:
            response = youtube.channels().list(part="snippet,statistics,contentDetails", forUsername=parts[1]).execute()
        else:
            raise ValueError("Unsupported channel URL. Use https://www.youtube.com/@handle or https://www.youtube.com/channel/CHANNEL_ID.")
    items = response.get("items", [])
    if not items:
        raise ValueError("No accessible public channel was found.")
    ch = items[0]
    snip, stats = ch.get("snippet", {}), ch.get("statistics", {})
    return {
        "channel_id": ch.get("id"),
        "channel_title": snip.get("title"),
        "channel_url": f"https://www.youtube.com/channel/{ch.get('id')}",
        "description": snip.get("description"),
        "published_at": snip.get("publishedAt"),
        "country": snip.get("country"),
        "subscriber_count": int(stats["subscriberCount"]) if "subscriberCount" in stats else None,
        "channel_view_count": int(stats["viewCount"]) if "viewCount" in stats else None,
        "channel_video_count": int(stats["videoCount"]) if "videoCount" in stats else None,
    }


def get_video(youtube, url):
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Could not extract a video ID. Provide a YouTube video URL.")
    response = youtube.videos().list(part="snippet,contentDetails,statistics", id=video_id).execute()
    items = response.get("items", [])
    if not items:
        raise ValueError("No accessible public video was found.")
    video = items[0]
    snip = video.get("snippet", {})
    content = video.get("contentDetails", {})
    stats = video.get("statistics", {})
    category_id = snip.get("categoryId")
    category = category_id
    if category_id:
        categories = youtube.videoCategories().list(part="snippet", regionCode="IN").execute()
        category = next((x["snippet"]["title"] for x in categories.get("items", []) if x["id"] == category_id), category_id)
    duration_iso = content.get("duration")
    duration_seconds = int(isodate.parse_duration(duration_iso).total_seconds()) if duration_iso else None
    channel = get_channel(youtube, f"https://www.youtube.com/channel/{snip.get('channelId')}")
    return {
        "video_id": video_id,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "title": snip.get("title"),
        "category": category,
        "category_id": category_id,
        "duration_seconds": duration_seconds,
        "view_count": int(stats["viewCount"]) if "viewCount" in stats else None,
        "like_count": int(stats["likeCount"]) if "likeCount" in stats else None,
        "comment_count": int(stats["commentCount"]) if "commentCount" in stats else None,
        **channel,
    }


def fetch_youtube_details(url: str) -> dict:
    """
    Fetch public details for a YouTube channel URL or video URL.

    VS Code test example:
        details = fetch_youtube_details("https://www.youtube.com/@Google")
        print(details)
    """
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "YOUTUBE_API_KEY is missing. Create a .env file beside this script "
            "with: YOUTUBE_API_KEY=your_actual_api_key"
        )

    youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
    try:
        if extract_video_id(url):
            return get_video(youtube, url)
        return get_channel(youtube, url)
    except (ValueError, HttpError) as exc:
        raise RuntimeError(f"Could not fetch YouTube details: {exc}") from exc


def test_youtube_details():
    """
    Simple manual test function for VS Code.
    Replace TEST_URL with a real YouTube channel or video URL.
    """
    TEST_URL = "https://www.youtube.com/@Google"
    details = fetch_youtube_details(TEST_URL)

    print("\nYouTube details fetched successfully:")
    for key, value in details.items():
        print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Fetch YouTube channel or video details.")
    parser.add_argument(
        "url",
        nargs="?",
        help="YouTube channel URL or video URL (optional; prompts if omitted)",
    )
    args = parser.parse_args()

    url = args.url or input("Paste a YouTube channel or video URL: ").strip()
    if not url:
        raise SystemExit("No URL entered.")

    try:
        details = fetch_youtube_details(url)
        print("\nYouTube details fetched successfully:")
        for key, value in details.items():
            print(f"{key}: {value}")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    # To test directly in VS Code, change TEST_URL in test_youtube_details()
    # and uncomment the next line:
    # test_youtube_details()
    main()
