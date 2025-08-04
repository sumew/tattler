FROM jrottenberg/ffmpeg:4.1-alpine

# This image comes with ffmpeg pre-installed.
# We just need to add Python and our script.

RUN apk add --no-cache python3

WORKDIR /app

COPY . .

# Create directories so docker-compose volumes mount correctly
RUN mkdir -p /app/videos
RUN mkdir -p /app/raw_frames

# The command is now passed to the python script
ENTRYPOINT ["python3", "main.py"]