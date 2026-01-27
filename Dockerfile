# Minimal Dockerfile for Flask + Gunicorn deployment
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5000

# Start the API using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "flask_app:app"]
