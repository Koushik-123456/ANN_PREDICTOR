FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Install system deps needed for some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501

# Use the PORT provided by the platform (Render sets $PORT)
CMD ["/bin/sh", "-c", "streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0"]
