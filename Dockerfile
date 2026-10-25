FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY flask_app/ /app/
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install -r requirements.txt
EXPOSE 5000

# local deploy
# CMD ["python","app.py"]
# production
CMD ["gunicorn","--bind","0.0.0.0:5000","--timeout","300","app:app"]

# to build the image
# docker build -t youtube_view_count_prediction:v1.0.0 .

# to run the image:
# docker run --rm   --name youtube-predictor   --env-file .env   -p 5000:5000 youtube_view_count_prediction:v1.0.0
