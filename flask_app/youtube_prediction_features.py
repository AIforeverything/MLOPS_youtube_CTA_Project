"""
Extract only YouTube video features needed for a view-count prediction model.
Model features: category, subscriber_count, channel_view_count, duration_seconds.
Current view count is returned separately for analysis, not model input.
"""
import argparse
import os
import re
from urllib.parse import urlparse, parse_qs

import certifi
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


def _youtube_client():
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY is missing. Add it to a .env file beside this script.")
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def _get_channel_features(youtube, channel_id):
    response = youtube.channels().list(part="snippet,statistics", id=channel_id).execute()
    items = response.get("items", [])
    if not items:
        raise ValueError("No accessible public channel was found.")
    stats = items[0].get("statistics", {})
    return {
        "subscriber_count": int(stats["subscriberCount"]) if "subscriberCount" in stats else None,
        "channel_view_count": int(stats["viewCount"]) if "viewCount" in stats else None,
    }


def extract_prediction_features(video_url):
    """Return model features and current view count from a video or Shorts URL."""
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL. Provide a video, Shorts, live, embed, or youtu.be URL.")

    youtube = _youtube_client()
    try:
        response = youtube.videos().list(
            part="snippet,contentDetails,statistics", id=video_id
        ).execute()
        items = response.get("items", [])
        if not items:
            raise ValueError("No accessible public video was found.")

        video = items[0]
        snippet = video.get("snippet", {})
        stats = video.get("statistics", {})
        content = video.get("contentDetails", {})

        category_id = snippet.get("categoryId")
        category = category_id
        if category_id:
            categories = youtube.videoCategories().list(part="snippet", regionCode="IN").execute()
            category = next(
                (item["snippet"]["title"] for item in categories.get("items", [])
                 if item["id"] == category_id),
                category_id,
            )

        duration = content.get("duration")
        duration_seconds = int(isodate.parse_duration(duration).total_seconds()) if duration else None
        channel_features = _get_channel_features(youtube, snippet["channelId"])

        model_features = {
            "category": category,
            "subscriber_count": channel_features["subscriber_count"],
            "channel_view_count": channel_features["channel_view_count"],
            "duration_seconds": duration_seconds,
        }
        return {
            "video_id": video_id,
            "model_features": model_features,
            "current_view_count": int(stats["viewCount"]) if "viewCount" in stats else None,
            "missing_features": [key for key, value in model_features.items() if value is None],
        }
    except HttpError as exc:
        raise RuntimeError(f"YouTube API request failed: {exc}") from exc


def test_extractor():
    """Manual VS Code test: replace TEST_URL with your own video/Shorts URL."""
    TEST_URL = "https://youtube.com/shorts/YuZGjOc_IYU"
    result = extract_prediction_features(TEST_URL)
    print("\nFeatures sent to the prediction model:")
    for key, value in result["model_features"].items():
        print(f"{key}: {value}")
    print(f"\nCurrent views (for analysis only): {result['current_view_count']}")
    if result["missing_features"]:
        print("Missing features:", ", ".join(result["missing_features"]))


def main():
    parser = argparse.ArgumentParser(description="Extract view-count prediction features from a YouTube URL.")
    parser.add_argument("url", nargs="?", help="YouTube video or Shorts URL")
    args = parser.parse_args()
    url = args.url or input("Paste a YouTube video or Shorts URL: ").strip()
    try:
        result = extract_prediction_features(url)
        print("\nFeatures sent to the prediction model:")
        for key, value in result["model_features"].items():
            print(f"{key}: {value}")
        print(f"\nCurrent views (for analysis only): {result['current_view_count']}")
        if result["missing_features"]:
            print("Missing features:", ", ".join(result["missing_features"]))
            print("Handle missing values before calling the model.")
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"Could not extract prediction features: {exc}") from exc


if __name__ == "__main__":
    main()
