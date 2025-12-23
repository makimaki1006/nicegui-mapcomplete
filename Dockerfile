# NiceGUI MapComplete Dashboard - Render用Dockerfile

FROM python:3.11-slim

# 作業ディレクトリ
WORKDIR /app

# 依存パッケージインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコード
COPY . .

# ポート設定（Renderは$PORTを使用）
ENV PORT=10000

# 起動コマンド
CMD ["python", "main.py"]
