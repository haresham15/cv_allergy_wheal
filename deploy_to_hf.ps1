# WhealVision Hugging Face Space Deployment Script
# Deploys backend to https://huggingface.co/spaces/hareshm15/allergy_wheal_api

param(
    [string]$Token = $env:HF_TOKEN
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " [Deploy] WhealVision to Hugging Face Gradio Space" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$RepoUrl = "https://huggingface.co/spaces/hareshm15/allergy_wheal_api"
$DeployDir = Join-Path $PSScriptRoot ".hf_space"
$BackendDir = Join-Path $PSScriptRoot "backend"

# 1. Get Access Token
if (-not $Token) {
    Write-Host "Please enter your Hugging Face Access Token (with WRITE permissions)." -ForegroundColor Yellow
    Write-Host "You can create or copy one here: https://huggingface.co/settings/tokens" -ForegroundColor Yellow
    Write-Host ""
    $Token = Read-Host -Prompt "Enter Hugging Face Token (starts with hf_)"
}

if (-not $Token -or -not $Token.Trim()) {
    Write-Error "Error: Hugging Face token is required to deploy."
    exit 1
}

$Token = $Token.Trim()
$AuthRepoUrl = "https://hareshm15:$Token@huggingface.co/spaces/hareshm15/allergy_wheal_api"

# 2. Clean or Prepare Deployment Directory
if (Test-Path $DeployDir) {
    Write-Host "[1/5] Cleaning existing staging directory..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $DeployDir
}

# 3. Clone the Space Repository
Write-Host "[2/5] Cloning Hugging Face Space repository..." -ForegroundColor Green
git clone $AuthRepoUrl $DeployDir
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to clone Space. Please check that your token has WRITE permissions." -ForegroundColor Red
    exit 1
}

# 4. Copy Production Backend Files
Write-Host "[3/5] Syncing production backend files..." -ForegroundColor Green

# Ensure directories exist in staging
$dirsToCreate = @("core", "routers", "services", "scripts", "models")
foreach ($d in $dirsToCreate) {
    $target = Join-Path $DeployDir $d
    if (-not (Test-Path $target)) {
        New-Item -ItemType Directory -Path $target | Out-Null
    }
}

# Copy root backend files
Copy-Item (Join-Path $BackendDir "app.py") -Destination $DeployDir -Force
Copy-Item (Join-Path $BackendDir "main.py") -Destination $DeployDir -Force
Copy-Item (Join-Path $BackendDir "requirements.txt") -Destination $DeployDir -Force
Copy-Item (Join-Path $BackendDir "packages.txt") -Destination $DeployDir -Force
Copy-Item (Join-Path $BackendDir "README.md") -Destination $DeployDir -Force
Copy-Item (Join-Path $BackendDir ".gitignore") -Destination $DeployDir -Force
Copy-Item (Join-Path $BackendDir "aruco_marker.png") -Destination $DeployDir -Force

# Copy package directories
Copy-Item -Path (Join-Path $BackendDir "core\*") -Destination (Join-Path $DeployDir "core") -Recurse -Force
Copy-Item -Path (Join-Path $BackendDir "routers\*") -Destination (Join-Path $DeployDir "routers") -Recurse -Force
Copy-Item -Path (Join-Path $BackendDir "services\*") -Destination (Join-Path $DeployDir "services") -Recurse -Force

# Copy scripts (omit pycache)
Copy-Item (Join-Path $BackendDir "scripts\download_sam.py") -Destination (Join-Path $DeployDir "scripts") -Force
Copy-Item (Join-Path $BackendDir "scripts\__init__.py") -Destination (Join-Path $DeployDir "scripts") -Force

# Copy models code (exclude heavy .pth weights)
Copy-Item (Join-Path $BackendDir "models\unet_rgbd.py") -Destination (Join-Path $DeployDir "models") -Force
Copy-Item (Join-Path $BackendDir "models\__init__.py") -Destination (Join-Path $DeployDir "models") -Force

# Clean up any __pycache__ in destination
Get-ChildItem -Path $DeployDir -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# 5. Commit and Push
Write-Host "[4/5] Committing and pushing to Hugging Face..." -ForegroundColor Green
Push-Location $DeployDir
try {
    git add .
    $status = git status --porcelain
    if ($status) {
        git commit -m "Deploy WhealVision Gradio and FastAPI backend with SAM and ZeroGPU support"
        git push origin main
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Git push failed. Please verify your token has WRITE permissions." -ForegroundColor Red
            exit 1
        }
        Write-Host "Successfully pushed to Hugging Face!" -ForegroundColor Green
    } else {
        Write-Host "Everything is already up-to-date in the Space!" -ForegroundColor Yellow
    }
} finally {
    # Sanitize remote URL to strip the access token from stored config
    git remote set-url origin $RepoUrl
    Pop-Location
}

# 6. Completion Summary
Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Deployment Complete!" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Your Space is now building and will be live at:"
Write-Host "  - Web Demo:         https://huggingface.co/spaces/hareshm15/allergy_wheal_api" -ForegroundColor Yellow
Write-Host "  - Direct REST API:  https://hareshm15-allergy-wheal-api.hf.space" -ForegroundColor Yellow
Write-Host "  - Health Check:     https://hareshm15-allergy-wheal-api.hf.space/health" -ForegroundColor Yellow
Write-Host "  - API Swagger Docs: https://hareshm15-allergy-wheal-api.hf.space/docs" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next Step:" -ForegroundColor Cyan
Write-Host "Once the space build is Complete (shows Running), test your live web app:"
Write-Host "  https://cv-allergy-wheal.vercel.app/" -ForegroundColor White
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""
