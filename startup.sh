#!/bin/bash
apt-get update && apt-get install -y ffmpeg
if [ -f antenv/bin/activate ]; then
  source antenv/bin/activate
else
  pip install -r requirements.txt
fi
gunicorn --bind=0.0.0.0 --timeout 600 app:app