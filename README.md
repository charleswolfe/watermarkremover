Install
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

540p?
python3 process_video_lama.py ~/Desktop/download\ 10.MP4 ~/Desktop/download\ 10-clean.MP4 ./ffmpeg '[{"x": 577, "y": 50, "w": 80, "h": 80},  {"x": 51, "y": 599, "w": 80, "h": 80},  {"x": 579, "y": 1148, "w": 80, "h": 80}, {"x": 494, "y": 58, "w": 85, "h": 38},  {"x": 128, "y": 609, "w": 85, "h": 38},  {"x": 496, "y": 1158, "w": 85, "h": 38}, {"x": 430, "y": 97, "w": 146, "h": 20},  {"x": 132, "y": 648, "w": 146, "h": 20},  {"x": 430, "y": 1197, "w": 148, "h": 20}]'

720p?
python3 process_video_lama.py ~/Desktop/bitch.MP4 ~/Desktop/bitch-clean.MP4 ./ffmpeg '[{"x": 31, "y": 108, "w": 63, "h": 63}, {"x": 103, "y": 122, "w": 88, "h": 37}, {"x": 506, "y": 602, "w": 63, "h": 63},  {"x": 586, "y": 619, "w": 88, "h": 37}, {"x": 31, "y": 1108, "w": 63, "h": 63}, {"x": 103, "y": 1124, "w": 88, "h": 37}]'


python3 process_video_lama.py ~/Desktop/download.mov ~/Desktop/downloads-clean.MP4 ./ffmpeg '[{"x": 31, "y": 108, "w": 63, "h": 63}, {"x": 103, "y": 122, "w": 88, "h": 37}, {"x": 506, "y": 602, "w": 63, "h": 63},  {"x": 586, "y": 619, "w": 88, "h": 37}, {"x": 31, "y": 1108, "w": 63, "h": 63}, {"x": 103, "y": 1124, "w": 88, "h": 37}]'

python3 process_video_lama.py ~/Desktop/download\ 2\ copy.MP4 ~/Desktop/download2copy-clean.MP4 ./ffmpeg
'[
{
    "x": 491,
    "y": 55,
    "w": 84,
    "h": 40
  },
  {
    "x": 579,
    "y": 49,
    "w": 74,
    "h": 81
  },
  {
    "x": 425,
    "y": 93,
    "w": 153,
    "h": 26
  },
  {
    "x": 52,
    "y": 601,
    "w": 76,
    "h": 80
  },
  {
    "x": 127,
    "y": 610,
    "w": 88,
    "h": 39
  },
  {
    "x": 128,
    "y": 647,
    "w": 150,
    "h": 24
  },
  {
    "x": 493,
    "y": 1162,
    "w": 80,
    "h": 31
  },
  {
    "x": 424,
    "y": 1198,
    "w": 149,
    "h": 21
  },
  {
    "x": 579,
    "y": 1155,
    "w": 72,
    "h": 72
  }
]'
Notes from cocde------

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
