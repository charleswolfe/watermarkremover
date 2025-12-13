Install


# mac:
install docker: `brew install docker docker-compose`
`brew install docker-credential-helper`

# Build the docker image
`docker build -f Dockerfile -t watermarkremover .`
`docker run -p 0.0.0.0:5000:5000  watermarkremover`
should be able to see page on http://localhost:5000/


------------------------------------------
delete me

# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Install required packages
pip install opencv-python numpy torch torchvision simple-lama-inpainting pillow


edit vi ./venv/lib/python3.13/site-packages/simple_lama_inpainting/models/model.py 
        self.model = torch.jit.load(model_path)
should be
        self.model = torch.jit.load(model_path, map_location='cpu')


call:
basic:
 python3 process_video_lama.py ~/Desktop/download\ 10.MP4 ~/Desktop/download\ 10-clean.MP4 ./ffmpeg '[{"x": 425, "y": 54, "w": 235, "h": 76}, {"x": 56, "y": 602, "w": 235, "h": 76}, {"x": 430, "y": 1150, "w": 235, "h": 76}]'
 
more specific:




# LaMa AI Model

This directory contains the big-lama.pt model file required for AI-powered watermark removal.

## Download

**File:** `big-lama.pt` (196 MB)

**Source:** https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt

**SHA256:** (To be verified on first download)

## Setup for Development

```bash
# Download the model
curl -L -o ./ \
  https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt

# Or if you already have it cached by PyTorch:
cp ~/.cache/torch/hub/checkpoints/big-lama.pt ./
```

## CI/CD Setup

For automated builds, ensure the model is downloaded before running PyInstaller:

```bash
# In your CI script
mkdir -p VideoFixer/Resources/lama
curl -L -o VideoFixer/Resources/lama/big-lama.pt \
  https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
```

## Why This Location?

The model must be in a **deterministic path** (not user cache) so that:
- ✅ CI/CD builds work without a cached model
- ✅ Notarization machines can build reproducibly
- ✅ Team members don't need to manually copy cache files

The PyInstaller spec (`process_video_lama_bundle.spec`) will **fail early** if the model is missing, preventing broken builds.

## Verification

After downloading, verify the model is embedded in the bundle:

```bash
# Build the bundle
pyinstaller process_video_lama_bundle.spec

# Check the model is included (should show big-lama.pt)
pyi-archive_viewer dist/process_video_lama_bundle | grep big-lama
```
