# build Angular frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .

RUN npm run build -- --configuration production

# Python backend
FROM python:3.11-slim AS backend
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install backend deps
COPY backend/requirements/requirements.txt ./backend/requirements/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements/requirements.txt

# Copy backend code
COPY backend/app ./backend/app

# Copy Angular build output into /app/backend/static
COPY --from=frontend-build /app/dist/frontend/browser /app/backend/static

# Expose and run
EXPOSE 5001
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "5001"]
