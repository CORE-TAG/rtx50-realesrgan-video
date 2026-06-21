# Publishing

The local repository is ready to publish. Creating the GitHub repository requires an authenticated GitHub account.

## Recommended GitHub CLI Flow

Install and authenticate GitHub CLI:

```powershell
winget install GitHub.cli
gh auth login
```

From the repository root:

```powershell
.\scripts\publish_github.ps1 -RepoName rtx50-realesrgan-video -Public
```

The script runs:

```powershell
gh repo create rtx50-realesrgan-video --public --source . --remote origin --push
gh repo edit --description "Windows RTX 50 RealESRGAN anime video upscaling pipeline, TensorRT/vs-mlrt benchmarks, and SM120 experiments"
```

## Manual Flow

Create an empty public GitHub repository named `rtx50-realesrgan-video`, then run:

```powershell
git remote add origin https://github.com/<owner>/rtx50-realesrgan-video.git
git push -u origin main
```

## Before Publishing

Confirm that these are not present:

- Videos or frame dumps
- TensorRT engines or timing caches
- ONNX/PTH/PT model files
- CUDA/TensorRT/FFmpeg binaries
- Virtual environments
- Private local paths or tokens

This prepared repository intentionally excludes all of those.

