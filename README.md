# MLOPS_youtube_Project
YouTube Video Performance Prediction and Optimization — An End-to-End MLOps Project

Objective:

Build an end-to-end MLOps system that analyzes 10,000 YouTube videos to understand the relationship between video duration, channel characteristics, content category, and audience engagement, predicts video performance, and provides data-driven recommendations for new videos.

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