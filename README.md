# MLOPS_youtube_CTA_Project
Predicting YouTube Subscription Call-to-Action Timing Using Channel, Video, and Transcript Features.
(when a YouTube creator will ask viewers to subscribe)

# 1. Clone repository

git clone <repository>

# 2. Install dependencies

pip install -r requirements.txt

# 3. Configure API key

cp .env.example .env

# Add:
YOUTUBE_API_KEY=xxxxx

# 4. Extract YouTube data

python scripts/collect_youtube_data.py

# 5. Extract transcripts

python scripts/collect_transcripts.py

# 6. Validate

python scripts/validate_data.py

# 7. Build features

python scripts/build_features.py

# 8. Train

python scripts/train.py