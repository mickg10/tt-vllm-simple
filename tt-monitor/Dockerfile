# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm install

# Copy source files
COPY frontend/ ./

# Build the frontend
RUN npm run build

# Stage 2: Runtime with Python backend
FROM ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-e95ffa5-48eba14

USER root

# Install Python dependencies for the monitor
WORKDIR /app

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Expose ports
# 9090: Dashboard and API
EXPOSE 9090

# Environment variables
ENV TT_MONITOR_MOCK=0
ENV FRONTEND_PATH=/app/frontend/dist

# Clear the base image entrypoint and run FastAPI server directly
ENTRYPOINT []
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "9090"]
