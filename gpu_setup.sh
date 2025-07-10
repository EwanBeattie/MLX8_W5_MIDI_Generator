python3 -m venv .venv

source .venv/bin/activate

apt update
apt install pip
apt install ffmpeg

pip install -r requirements.txt