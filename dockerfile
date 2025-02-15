
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "main1.py", "--server.port", "8501", "--server.enableCORS", "false"]
