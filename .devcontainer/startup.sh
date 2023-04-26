#!/bin/sh
sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y
python3 -m pip install --user -r requirements.txt

echo "alias ll='ls -lh'" > ~/.bash_aliases