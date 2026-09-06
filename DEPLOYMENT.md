# Deployment Guide: Allergy Wheal Detection App

This guide outlines the production deployment strategy for the Allergy Wheal Detection Tracker. Because the backend utilizes deep learning models (PyTorch + Meta's Segment Anything Model), it requires at least 2–4 GB RAM for inference.

The recommended deployment topology:
- **Frontend (Next.js):** Hosted on **Vercel** (zero-config, free global CDN).
- **Backend (FastAPI + SAM):** Hosted on **Hugging Face Spaces** (free tier with **16 GB RAM & 2 vCPUs**) or **Google Cloud Run / AWS / VPS** via Docker.
- **Local / Self-Hosted:** Managed with **Docker Compose** (`docker compose up --build`).

---

## 1. Local / VPS Deployment with Docker Compose

To run the complete production stack (Backend + Frontend) locally or on a VPS:

1. Clone the repository and navigate to the project root:
   ```bash
   git clone <repo-url>
   cd cv_allergy_wheal
   ```
2. Start the services using Docker Compose:
   ```bash
   docker compose up --build
   ```
3. Access the applications:
   - **Frontend Web UI:** `http://localhost:3000`
   - **FastAPI Backend API:** `http://localhost:8000`
   - **Interactive API Docs (Swagger):** `http://localhost:8000/docs`
   - **Health Check:** `http://localhost:8000/health`

---

## 2. Backend Deployment (Hugging Face Spaces - Free 16GB RAM Tier)

Hugging Face Spaces provides **16 GB RAM and 2 vCPUs** on its free tier, making it ideal for hosting the SAM ViT-B model without out-of-memory errors.

### Steps:
1. Sign in or create a free account at [Hugging Face](https://huggingface.co/).
2. Create a **New Space**:
   - **Space Name:** `allergy-wheal-api` (or your choice)
   - **License:** MIT
   - **Space SDK:** Select **Docker**
   - **Docker Template:** Select **Blank**
   - **Hardware:** Free tier (2 vCPU, 16 GB RAM)
3. Clone your new Hugging Face Space repository locally or upload files:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/allergy-wheal-api
   ```
4. Copy all contents of the `backend/` directory into your Space repository:
   - `Dockerfile`
   - `.dockerignore`
   - `requirements.txt`
   - `main.py`
   - `core/`
   - `routers/`
   - `services/`
   - `scripts/`
   - `models/`
5. Commit and push:
   ```bash
   git add .
   git commit -m "Deploy Allergy Wheal Detection API"
   git push origin main
   ```
6. Hugging Face will automatically build the Docker image (downloading the SAM ViT-B model during build) and launch the container on port `7860`.
7. Once the build finishes, your API URL will be:
   `https://<your-username>-allergy-wheal-api.hf.space`
8. Verify the deployment:
   ```bash
   curl https://<your-username>-allergy-wheal-api.hf.space/health
   ```

---

## 3. Frontend Deployment (Vercel)

Vercel provides seamless zero-configuration hosting for the Next.js frontend with automatic SSL and global CDN distribution.

### Steps:
1. Ensure your repository is pushed to GitHub, GitLab, or Bitbucket.
2. Log into [Vercel](https://vercel.com/) and click **Add New... > Project**.
3. Import your Git repository.
4. **Configure Project Settings:**
   - **Root Directory:** Click `Edit` and select `frontend/cv_allergy_wheal`.
   - **Framework Preset:** Next.js (detected automatically).
5. **Set Environment Variables:**
   - Add `NEXT_PUBLIC_API_URL` with the URL of your deployed backend:
     `NEXT_PUBLIC_API_URL = https://<your-username>-allergy-wheal-api.hf.space`
   - (Optional) Add `BACKEND_API_URL` to route requests through the Next.js `/api/analyze` proxy route to eliminate cross-origin issues:
     `BACKEND_API_URL = https://<your-username>-allergy-wheal-api.hf.space`
6. Click **Deploy**.
7. Vercel will build the frontend and provide your live production URL (e.g., `https://cv-allergy-wheal.vercel.app`).

---

## 4. Alternative: Google Cloud Run Deployment

If deploying to Google Cloud Run:
```bash
cd backend
gcloud run deploy allergy-wheal-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 120s
```
*Note: Ensure memory is configured to at least 4Gi for SAM ViT-B.*

---

## 5. Environment Variables Reference

### Backend (`backend/.env`):
| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port for the Uvicorn web server |
| `CORS_ORIGINS` | `*` | Comma-separated list of allowed origins or `*` |

### Frontend (`frontend/cv_allergy_wheal/.env`):
| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Public backend URL accessed by browser or build |
| `BACKEND_API_URL` | `http://localhost:8000` | Internal/server backend URL used by Next.js API proxy |
