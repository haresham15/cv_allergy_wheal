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

## 2. Backend Deployment (Hugging Face Spaces - Free Tier)

Hugging Face Spaces offers **Free Gradio Spaces** with **ZeroGPU (Nvidia A100)** or **CPU Basic (16 GB RAM)**. Custom Docker Spaces require a paid subscription, but the **Gradio SDK** is completely free and natively runs our FastAPI application alongside an interactive UI!

### Why Gradio SDK?

- **100% Free**: No subscription required.
- **FastAPI Mount**: Our `app.py` mounts the FastAPI application onto Gradio, so all REST endpoints (`/api/v1/analyze`, `/health`, `/docs`) work directly for your Next.js frontend!
- **Free GPU Acceleration**: Choosing **ZeroGPU (Free)** gives free Nvidia A100 GPU compute during inference, reducing analysis time to under 1 second!

### Quick 1-Click Deployment (PowerShell)

We have created an automated deployment script `deploy_to_hf.ps1` in the project root that automatically clones your Space, syncs the backend files, and pushes to Hugging Face:

```powershell
.\deploy_to_hf.ps1
```

*(It will prompt for your Hugging Face Access Token with **WRITE** permissions from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).*

---

### Manual Steps

1. Ensure your Space at [huggingface.co/spaces/hareshm15/allergy_wheal_api](https://huggingface.co/spaces/hareshm15/allergy_wheal_api) is set to **Public** in Space Settings (so the Next.js frontend on Vercel can reach the API without 401 Unauthorized errors).
2. Clone your Hugging Face Space repository:

   ```bash
   git clone https://huggingface.co/spaces/hareshm15/allergy_wheal_api
   ```

3. Copy all files from `backend/` into `allergy_wheal_api/`:
   - `app.py` (entry point for Gradio + FastAPI)
   - `packages.txt` (installs system libraries `libgl1`, `libglib2.0-0` automatically)
   - `requirements.txt`
   - `README.md`
   - `main.py`
   - `core/`
   - `routers/`
   - `services/`
   - `scripts/` (omits large data/caches, includes `download_sam.py`)
   - `models/` (omits the 375MB `.pth` file; `app.py` auto-downloads it on boot)
4. Commit and push:

   ```bash
   git add .
   git commit -m "Deploy WhealVision Backend"
   git push origin main
   ```

5. Hugging Face will automatically install `packages.txt`, `requirements.txt`, launch `app.py`, and expose port `7860`.
6. Your live public API URL will be:
   `https://hareshm15-allergy-wheal-api.hf.space`
7. Test the live endpoints:
   - **Health Check:** `curl https://hareshm15-allergy-wheal-api.hf.space/health`
   - **API Docs:** Visit `https://hareshm15-allergy-wheal-api.hf.space/docs` in your browser
   - **Interactive Web UI:** Visit `https://hareshm15-allergy-wheal-api.hf.space/` to test image uploads directly!

---

## 3. Frontend Deployment (Vercel)

Vercel provides seamless zero-configuration hosting for the Next.js frontend with automatic SSL and global CDN distribution.

### Steps

1. Ensure your repository is pushed to GitHub, GitLab, or Bitbucket.
2. Log into [Vercel](https://vercel.com/) and click **Add New... > Project**.
3. Import your Git repository.
4. **Configure Project Settings:**
   - **Root Directory:** Click `Edit` and select `frontend/cv_allergy_wheal`.
   - **Framework Preset:** Next.js (detected automatically).
5. **Set Environment Variables:**
   - Add `NEXT_PUBLIC_API_URL` with the URL of your deployed backend:
     `NEXT_PUBLIC_API_URL = https://hareshm15-allergy-wheal-api.hf.space`
   - (Optional) Add `BACKEND_API_URL` to route requests through the Next.js `/api/analyze` proxy route to eliminate cross-origin issues:
     `BACKEND_API_URL = https://hareshm15-allergy-wheal-api.hf.space`
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

### Backend (`backend/.env`)

| Variable | Default | Description |
| --- | --- | --- |
| `PORT` | `8000` | Port for the Uvicorn web server |
| `CORS_ORIGINS` | `*` | Comma-separated list of allowed origins or `*` |

### Frontend (`frontend/cv_allergy_wheal/.env`)

| Variable | Default | Description |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Public backend URL accessed by browser or build |
| `BACKEND_API_URL` | `http://localhost:8000` | Internal/server backend URL used by Next.js API proxy |
