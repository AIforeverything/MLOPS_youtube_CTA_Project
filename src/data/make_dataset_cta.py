import os
import time
import isodate
import pandas as pd
import re

from dotenv import load_dotenv
from googleapiclient.discovery import build

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None


# =========================================================
# CONFIGURATION
# =========================================================

# Required packages:
# python -m pip install google-api-python-client python-dotenv isodate pandas youtube-transcript-api


load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found in .env")

youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)

CATEGORIES = {
    "Technology": "technology",
    "Education": "education",
    "Gaming": "gaming",
    "Entertainment": "entertainment",
    "Science": "science",
    "Fitness": "fitness",
    "Finance": "finance",
    "Travel": "travel",
    "Food": "food",
    "News": "news"
}

CHANNELS_PER_CATEGORY = 50
VIDEOS_PER_CHANNEL = 20


# =========================================================
# 1. DISCOVER CHANNELS
# =========================================================

def discover_channels(query, limit=50):
    """Discover unique YouTube channels for a search query."""

    channels = {}
    next_page_token = None

    while len(channels) < limit:
        try:
            response = youtube.search().list(
                part="snippet",
                q=query,
                type="channel",
                maxResults=50,
                pageToken=next_page_token
            ).execute()

        except Exception as e:
            print(f"WARNING: Channel discovery failed for '{query}'")
            print(f"Reason: {e}")
            break

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            channel_id = snippet.get("channelId")

            if not channel_id:
                continue

            if channel_id not in channels:
                channels[channel_id] = {
                    "channel_id": channel_id,
                    "channel_name": snippet.get("title")
                }

            if len(channels) >= limit:
                break

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

        time.sleep(0.1)

    return list(channels.values())


# =========================================================
# 2. GET CHANNEL DETAILS
# =========================================================

def get_channel_details(channel_ids):
    """
    Get channel statistics and the official uploads playlist ID.

    The uploads playlist ID is obtained directly from the
    YouTube API. It is not constructed manually.
    """

    results = []

    for start in range(0, len(channel_ids), 50):
        batch = channel_ids[start:start + 50]

        try:
            response = youtube.channels().list(
                part="snippet,statistics,contentDetails",
                id=",".join(batch)
            ).execute()

        except Exception as e:
            print("WARNING: Could not retrieve channel batch.")
            print(f"Reason: {e}")
            continue

        for item in response.get("items", []):
            statistics = item.get("statistics", {})
            snippet = item.get("snippet", {})
            content_details = item.get("contentDetails", {})

            related_playlists = content_details.get(
                "relatedPlaylists", {}
            )

            uploads_playlist = related_playlists.get("uploads")

            if not uploads_playlist:
                print(
                    f"WARNING: No uploads playlist "
                    f"for channel {item.get('id')}"
                )

            results.append({
                "channel_id": item.get("id"),
                "channel_name": snippet.get("title"),
                "subscriber_count": int(
                    statistics.get("subscriberCount", 0)
                ),
                "channel_view_count": int(
                    statistics.get("viewCount", 0)
                ),
                "video_count": int(
                    statistics.get("videoCount", 0)
                ),
                "uploads_playlist": uploads_playlist
            })

    return results


# =========================================================
# 3. GET MOST RECENT VIDEOS
# =========================================================

def get_recent_videos(uploads_playlist, limit=20):
    """
    Retrieve the most recent videos from a channel's
    uploads playlist.

    Invalid/deleted/inaccessible playlists are skipped
    instead of stopping the complete pipeline.
    """

    videos = []

    if not uploads_playlist:
        print("  WARNING: Uploads playlist ID is missing")
        return videos

    next_page_token = None

    while len(videos) < limit:
        try:
            response = youtube.playlistItems().list(
                part="contentDetails,snippet",
                playlistId=uploads_playlist,
                maxResults=min(50, limit - len(videos)),
                pageToken=next_page_token
            ).execute()

        except Exception as e:
            print(
                f"  WARNING: Could not retrieve "
                f"playlist: {uploads_playlist}"
            )
            print(f"  Reason: {e}")
            return []

        for item in response.get("items", []):
            content_details = item.get("contentDetails", {})

            video_id = content_details.get("videoId")

            if not video_id:
                continue

            published_at = content_details.get("videoPublishedAt")

            if not published_at:
                published_at = (
                    item.get("snippet", {})
                    .get("publishedAt")
                )

            videos.append({
                "video_id": video_id,
                "published_at": published_at
            })

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    return videos[:limit]


# =========================================================
# 4. GET VIDEO DETAILS
# =========================================================

def get_video_details(video_ids):
    """Get metadata, duration and statistics for videos."""

    results = []

    if not video_ids:
        return results

    for start in range(0, len(video_ids), 50):
        batch = video_ids[start:start + 50]

        try:
            response = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=",".join(batch)
            ).execute()

        except Exception as e:
            print("WARNING: Could not retrieve video details.")
            print(f"Reason: {e}")
            continue

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            content_details = item.get("contentDetails", {})
            statistics = item.get("statistics", {})

            duration = content_details.get("duration")

            if duration:
                try:
                    duration_seconds = int(
                        isodate.parse_duration(
                            duration
                        ).total_seconds()
                    )
                except Exception:
                    duration_seconds = None
            else:
                duration_seconds = None

            results.append({
                "video_id": item.get("id"),
                "video_title": snippet.get("title"),
                "published_at": snippet.get("publishedAt"),
                "description": snippet.get("description"),
                "duration_seconds": duration_seconds,
                "view_count": int(
                    statistics.get("viewCount", 0)
                ),
                "like_count": int(
                    statistics.get("likeCount", 0)
                ),
                "comment_count": int(
                    statistics.get("commentCount", 0)
                )
            })

    return results


# =========================================================
# 5. MAIN DATA COLLECTION PIPELINE
# =========================================================

def main():

    all_rows = []

    for category, search_query in CATEGORIES.items():

        print("\n" + "=" * 60)
        print(f"CATEGORY: {category}")
        print("=" * 60)

        # -------------------------------------------------
        # Discover channels
        # -------------------------------------------------

        print("Discovering channels...")

        channels = discover_channels(
            search_query,
            CHANNELS_PER_CATEGORY
        )

        print(
            f"Channels discovered: {len(channels)}"
        )

        if not channels:
            print(
                "No channels discovered - "
                "moving to next category."
            )
            continue

        # -------------------------------------------------
        # Get channel statistics
        # -------------------------------------------------

        channel_ids = [
            c["channel_id"]
            for c in channels
        ]

        channel_details = get_channel_details(channel_ids)

        print(
            f"Channel details retrieved: "
            f"{len(channel_details)}"
        )

        # -------------------------------------------------
        # Process every channel
        # -------------------------------------------------

        for channel_number, channel in enumerate(
            channel_details,
            start=1
        ):

            try:
                channel_id = channel.get("channel_id")
                channel_name = channel.get(
                    "channel_name",
                    "Unknown"
                )

                print(
                    f"[{channel_number}/"
                    f"{len(channel_details)}] "
                    f"{channel_name}"
                )

                uploads_playlist = channel.get(
                    "uploads_playlist"
                )

                if not uploads_playlist:
                    print(
                        "  No uploads playlist - skipping"
                    )
                    continue

                print(
                    f"  Uploads playlist: "
                    f"{uploads_playlist}"
                )

                # -----------------------------------------
                # Get recent videos
                # -----------------------------------------

                videos = get_recent_videos(
                    uploads_playlist,
                    VIDEOS_PER_CHANNEL
                )

                if not videos:
                    print(
                        "  Could not retrieve recent "
                        "videos - skipping"
                    )
                    continue

                print(
                    f"  Recent videos found: {len(videos)}"
                )

                video_ids = [
                    v["video_id"]
                    for v in videos
                ]

                # -----------------------------------------
                # Get video metadata
                # -----------------------------------------

                video_details = get_video_details(video_ids)

                if not video_details:
                    print(
                        "  No video details found - "
                        "skipping"
                    )
                    continue

                # -----------------------------------------
                # Combine channel + video data
                # -----------------------------------------

                for video in video_details:

                    row = {
                        "category": category,
                        "channel_id": channel_id,
                        "channel_name": channel_name,
                        "subscriber_count": channel[
                            "subscriber_count"
                        ],
                        "channel_view_count": channel[
                            "channel_view_count"
                        ],
                        "channel_video_count": channel[
                            "video_count"
                        ],
                        "video_id": video["video_id"],
                        "video_title": video["video_title"],
                        "published_at": video["published_at"],
                        "duration_seconds": video[
                            "duration_seconds"
                        ],
                        "view_count": video["view_count"],
                        "like_count": video["like_count"],
                        "comment_count": video["comment_count"],
                        "description": video["description"],
                        "video_url": (
                            "https://www.youtube.com/watch?v="
                            + video["video_id"]
                        )
                    }

                    all_rows.append(row)

            except Exception as e:
                print(
                    f"  ERROR processing channel "
                    f"{channel.get('channel_name', 'Unknown')}"
                )
                print(f"  Reason: {e}")
                continue

    # =====================================================
    # CREATE DATAFRAME
    # =====================================================

    if not all_rows:
        print(
            "\nNo data collected. Nothing to save."
        )
        return

    df = pd.DataFrame(all_rows)

    # Remove duplicate videos
    df = df.drop_duplicates(
        subset=["video_id"]
    )

    # Sort
    df = df.sort_values(
        [
            "category",
            "channel_name",
            "published_at"
        ]
    )

    # =====================================================
    # SAVE DATA
    # =====================================================

    output_dir = "data/raw"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        output_dir,
        "youtube_10000_videos.csv"
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)

    print(
        f"File saved: {output_file}"
    )

    print(
        f"Total unique videos: {len(df)}"
    )

    print(
        f"Total channels: "
        f"{df['channel_id'].nunique()}"
    )

    print(
        f"Total categories: "
        f"{df['category'].nunique()}"
    )

    print("\nVideos per category:")

    print(
        df.groupby("category").size()
    )


# =========================================================
# 6. CTA / TRANSCRIPT EXTRACTION
# =========================================================

# These patterns are intentionally focused on subscription-related
# calls-to-action rather than every occurrence of the word "subscribe".
CTA_PATTERNS = [
    r"\bplease\s+(?:hit|press|click|tap|smash)\s+(?:the\s+)?subscribe\b",
    r"\b(?:hit|press|click|tap|smash)\s+(?:the\s+)?subscribe\s+button\b",
    r"\b(?:don't|do not)\s+forget\s+to\s+subscribe\b",
    r"\bmake\s+sure\s+(?:you\s+)?(?:to\s+)?subscribe\b",
    r"\bmake\s+sure\s+you(?:'ve| have)?\s+subscribed\b",
    r"\b(?:please\s+)?subscribe\s+to\s+(?:my|our|the)\s+channel\b",
    r"\b(?:please\s+)?subscribe\s+to\s+(?:this|my|our)\s+channel\b",
    r"\bconsider\s+subscribing\b",
    r"\bsubscribe\s+for\s+more\b",
    r"\bif\s+you(?:'re| are)\s+enjoying.*subscribe\b",
    r"\b(?:like|liked)\s+the\s+video.*subscribe\b",
    r"\b(?:don't|do not)\s+forget.*subscribe\b",
    r"\b(?:be\s+sure|sure)\s+to\s+subscribe\b",
    r"\b(?:go\s+ahead\s+and\s+)?subscribe\b",
]


def is_subscribe_cta(text):
    """
    Return True when transcript text looks like a subscription CTA.

    A simple standalone "subscribe" is also accepted because many
    creators say phrases such as "subscribe and turn on notifications".
    """
    if not text:
        return False

    normalized = re.sub(r"\s+", " ", text.lower()).strip()

    for pattern in CTA_PATTERNS:
        if re.search(pattern, normalized):
            return True

    # Common short CTA forms that may not match the longer patterns.
    short_patterns = [
        r"\bsubscribe\b.*\b(?:notification|notifications|bell|channel)\b",
        r"\b(?:subscribe|subscribed)\b.*\b(?:channel|youtube)\b",
    ]

    return any(
        re.search(pattern, normalized)
        for pattern in short_patterns
    )


def get_transcript(video_id):
    """
    Fetch a video's English transcript.

    youtube-transcript-api is independent of the YouTube Data API
    quota. The current library API uses:
        YouTubeTranscriptApi().fetch(video_id)

    Returns a list of timestamped transcript snippets.
    """
    if YouTubeTranscriptApi is None:
        raise ImportError(
            "youtube-transcript-api is not installed. "
            "Run: python -m pip install youtube-transcript-api"
        )

    api = YouTubeTranscriptApi()

    # Try English first, then Hindi/English fallback if available.
    try:
        transcript = api.fetch(
            video_id,
            languages=["en"]
        )
    except Exception:
        try:
            transcript = api.fetch(
                video_id,
                languages=["en", "hi"]
            )
        except Exception as e:
            raise RuntimeError(str(e))

    # Newer versions return FetchedTranscript objects.
    # Convert them to a simple list of dictionaries.
    snippets = []

    for snippet in transcript:
        text = getattr(snippet, "text", "")
        start = getattr(snippet, "start", 0.0)
        duration = getattr(snippet, "duration", 0.0)

        snippets.append({
            "text": text,
            "start": float(start),
            "duration": float(duration)
        })

    return snippets


def extract_cta_from_transcript(
    video_id,
    duration_seconds
):
    """
    Find subscription CTA occurrences and calculate the first CTA position.

    Returns:
        subscribe_count
        first_subscribe_timestamp
        first_subscribe_percentage
        subscribe_text
        transcript_status
    """

    result = {
        "subscribe_count": 0,
        "first_subscribe_timestamp": None,
        "first_subscribe_percentage": None,
        "subscribe_text": None,
        "transcript_status": "not_checked"
    }

    if not video_id:
        result["transcript_status"] = "missing_video_id"
        return result

    try:
        snippets = get_transcript(video_id)
        result["transcript_status"] = "available"
    except Exception as e:
        result["transcript_status"] = f"unavailable: {str(e)[:150]}"
        return result

    cta_matches = []

    for snippet in snippets:
        text = snippet["text"]

        if is_subscribe_cta(text):
            cta_matches.append({
                "start": snippet["start"],
                "text": text
            })

    result["subscribe_count"] = len(cta_matches)

    if cta_matches:
        first_cta = cta_matches[0]

        result["first_subscribe_timestamp"] = round(
            first_cta["start"],
            2
        )

        result["subscribe_text"] = first_cta["text"]

        if duration_seconds is not None:
            try:
                duration = float(duration_seconds)

                if duration > 0:
                    percentage = (
                        first_cta["start"] /
                        duration
                    ) * 100

                    result["first_subscribe_percentage"] = round(
                        min(max(percentage, 0), 100),
                        2
                    )
            except (TypeError, ValueError):
                pass

    return result


def enrich_csv_with_cta(
    input_file="data/raw/youtube_10000_videos.csv",
    output_file="data/interim/youtube_cta_dataset.csv",
    update_raw_file=False
):
    """
    Read the existing YouTube metadata CSV and add CTA columns.

    This does NOT call the YouTube Data API again.

    By default:
        input  = data/raw/youtube_10000_videos.csv
        output = data/interim/youtube_cta_dataset.csv

    If update_raw_file=True, the same CTA columns are also written
    back to the original raw CSV.
    """

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"CSV file not found: {input_file}"
        )

    print("\n" + "=" * 60)
    print("CTA EXTRACTION FROM EXISTING CSV")
    print("=" * 60)

    df = pd.read_csv(input_file)

    required_columns = [
        "video_id",
        "duration_seconds"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns missing from CSV: {missing}"
        )

    # Existing CTA columns allow the script to be resumed.
    cta_columns = [
        "subscribe_count",
        "first_subscribe_timestamp",
        "first_subscribe_percentage",
        "subscribe_text",
        "transcript_status"
    ]

    for column in cta_columns:
        if column not in df.columns:
            df[column] = None

    total = len(df)

    print(f"Videos in CSV: {total}")
    print("Starting transcript extraction...\n")

    for index, row in df.iterrows():

        video_id = str(row["video_id"]).strip()

        if not video_id or video_id.lower() == "nan":
            continue

        # Resume support:
        # skip a video if CTA/transcript information is already present.
        status = row.get("transcript_status")

        if pd.notna(status) and str(status).strip() != "":
            print(
                f"[{index + 1}/{total}] "
                f"{video_id} -> already processed"
            )
            continue

        print(
            f"[{index + 1}/{total}] "
            f"Processing {video_id}"
        )

        result = extract_cta_from_transcript(
            video_id=video_id,
            duration_seconds=row["duration_seconds"]
        )

        for column, value in result.items():
            df.at[index, column] = value

        if result["subscribe_count"] > 0:
            print(
                f"  CTA found: "
                f"{result['subscribe_count']} | "
                f"first at "
                f"{result['first_subscribe_timestamp']} sec | "
                f"{result['first_subscribe_percentage']}%"
            )
            print(
                f"  Text: {result['subscribe_text']}"
            )
        else:
            print(
                f"  No subscription CTA found "
                f"({result['transcript_status']})"
            )

        # Small delay to avoid hammering the transcript endpoint.
        time.sleep(0.2)

        # Save periodically so a long run can be resumed.
        if (index + 1) % 25 == 0:
            os.makedirs(
                os.path.dirname(output_file),
                exist_ok=True
            )

            df.to_csv(
                output_file,
                index=False,
                encoding="utf-8"
            )

            print(
                f"  Checkpoint saved: {output_file}"
            )

    # ---------------------------------------------------------
    # Save final enriched dataset
    # ---------------------------------------------------------

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    # Optional: write CTA columns back into the original raw CSV.
    if update_raw_file:
        df.to_csv(
            input_file,
            index=False,
            encoding="utf-8"
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    transcript_available = (
        df["transcript_status"]
        .astype(str)
        .str.startswith("available")
    )

    cta_found = (
        pd.to_numeric(
            df["subscribe_count"],
            errors="coerce"
        )
        .fillna(0)
        > 0
    )

    print("\n" + "=" * 60)
    print("CTA EXTRACTION COMPLETE")
    print("=" * 60)

    print(f"Output file: {output_file}")
    print(f"Total videos: {len(df)}")
    print(
        f"Transcripts available: "
        f"{transcript_available.sum()}"
    )
    print(
        f"Videos with CTA detected: "
        f"{cta_found.sum()}"
    )
    print(
        f"Videos without CTA detected: "
        f"{(~cta_found).sum()}"
    )

    if cta_found.any():
        print("\nCTA statistics:")
        print(
            df.loc[
                cta_found,
                [
                    "subscribe_count",
                    "first_subscribe_timestamp",
                    "first_subscribe_percentage"
                ]
            ].describe()
        )

    print("\nFirst few CTA records:")

    display_columns = [
        "video_id",
        "video_title",
        "duration_seconds",
        "subscribe_count",
        "first_subscribe_timestamp",
        "first_subscribe_percentage",
        "subscribe_text"
    ]

    print(
        df.loc[
            cta_found,
            display_columns
        ].head(10).to_string(index=False)
    )

    return df


# =========================================================
# 7. COMPLETE PIPELINE
# =========================================================

def run_pipeline():
    """
    Complete project pipeline.

    If the raw CSV already exists, CTA extraction is performed
    directly on that CSV.

    If the raw CSV does not exist, the YouTube Data API collection
    pipeline runs first and then CTA extraction starts.
    """

    raw_file = "data/raw/youtube_10000_videos.csv"

    processed_file = (
        "data/interim/youtube_cta_dataset.csv"
    )

    if os.path.exists(raw_file):
        print(
            f"Existing dataset found: {raw_file}"
        )
        print(
            "Skipping YouTube metadata collection."
        )

    else:
        print(
            "Raw CSV does not exist."
        )
        print(
            "Running YouTube metadata collection first..."
        )

        main()

    # Add CTA information to the existing CSV.
    enrich_csv_with_cta(
        input_file=raw_file,
        output_file=processed_file,
        update_raw_file=False
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    run_pipeline()
