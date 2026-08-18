# ベースイメージとして Python 3.12 (Debian Bookwormベースで脆弱性を抑制) を使用
FROM python:3.12-slim-bookworm@sha256:a9f9f61f7e4b0a7e5c3e5b5c5e5f5f5f5f5f5f5f5f5f5f5f5f5f5f5f5f5f5f

# 作業ディレクトリの設定
WORKDIR /app

# システムの依存パッケージ（GUIや音声処理に必要なライブラリ）のインストール
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    libgl1-mesa-glx \
    libegl1 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libx11-xcb1 \
    libfontconfig1 \
    libdbus-1-3 \
    libasound2 \
    libportaudio2 \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# 依存パッケージのコピーとインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードのコピー
COPY . .

# 実行コマンド
CMD ["python", "main.py"]
