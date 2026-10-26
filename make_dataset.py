import os
import time
import isodate
import pandas as pd

from dotenv import load_dotenv
from googleapiclient.discovery import build


# =========================================================
# CONFIGURATION
# =========================================================

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
# RUN
# =========================================================

if __name__ == "__main__":
    main()
