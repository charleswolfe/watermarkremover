#!/usr/bin/env python3
"""
AI-based watermark removal using LaMa (Large Mask Inpainting)
Uses FFmpeg for frame extraction and re-encoding
GPU-accelerated for M-series Macs

Supports two modes:
  - Single video mode: process_video_lama.py <input> <output> <ffmpeg> <regions>
  - Batch mode: process_video_lama.py --batch <ffmpeg>
    (reads video requests from stdin, one per line: "input|output|regions")
"""

import cv2
import numpy as np
import subprocess
import os
import sys
import tempfile
import torch
from simple_lama_inpainting import SimpleLama
from PIL import Image
import ssl
import json
import traceback
import signal
import shutil
import time
import contextlib

# Disable SSL verification for model download
# This is a one-time download during the app's lifetime
ssl._create_default_https_context = ssl._create_unverified_context


class CancellationError(Exception):
    """Raised when processing is cancelled by the user."""


MODE = "unknown"  # Either "single" or "batch" once main() dispatches
CANCELLED = False
ACTIVE_PROCESSES = set()
PROCESS_POLL_INTERVAL = 0.1
PROCESS_TERMINATE_TIMEOUT = 3.0


def register_subprocess(proc):
    """Track spawned subprocesses so they can be terminated on cancel."""
    ACTIVE_PROCESSES.add(proc)


def deregister_subprocess(proc):
    ACTIVE_PROCESSES.discard(proc)


def terminate_active_subprocesses(force=False):
    """Terminate or kill all tracked subprocesses, including their process groups."""
    if not ACTIVE_PROCESSES:
        return

    procs = list(ACTIVE_PROCESSES)
    for proc in procs:
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            deregister_subprocess(proc)
            continue

        try:
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.killpg(pgid, sig)
        except ProcessLookupError:
            deregister_subprocess(proc)

    deadline = time.time() + PROCESS_TERMINATE_TIMEOUT
    remaining = list(ACTIVE_PROCESSES)
    while remaining and time.time() < deadline:
        for proc in list(remaining):
            if proc.poll() is not None:
                deregister_subprocess(proc)
                remaining.remove(proc)
        time.sleep(PROCESS_POLL_INTERVAL)

    if remaining:
        # Force kill anything still alive
        for proc in remaining:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                try:
                    proc.wait(timeout=1)
                except Exception:
                    pass
                deregister_subprocess(proc)


def raise_if_cancelled():
    if CANCELLED:
        raise CancellationError("Processing cancelled")


def request_cancel(reason: str):
    """Mark cancellation and tear down subprocesses."""
    global CANCELLED
    if CANCELLED:
        return

    CANCELLED = True
    terminate_active_subprocesses(force=False)

    prefix = "BATCH|CANCELLED" if MODE == "batch" else "CANCELLED"
    print(f"{prefix}|reason={reason}", file=sys.stderr, flush=True)

def run_monitored_subprocess(cmd):
    """Launch a subprocess while tracking for cooperative cancellation."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    register_subprocess(proc)

    try:
        stdout, stderr = proc.communicate()
    finally:
        deregister_subprocess(proc)

    if CANCELLED:
        raise CancellationError("Processing cancelled")

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            cmd,
            output=stdout,
            stderr=stderr,
        )

    return stdout, stderr


def extract_frames(video_path, output_dir, ffmpeg_path):
    """Extract frames from video using FFmpeg"""
    cmd = [
        ffmpeg_path,
        '-i', video_path,
        '-qscale:v', '2',  # High quality
        f'{output_dir}/frame_%06d.png'
    ]
    run_monitored_subprocess(cmd)

def remove_watermark_lama(simple_lama, frame_path, output_path, regions):
    """Remove watermarks using LaMa AI inpainting"""
    raise_if_cancelled()
    # Read image
    image = cv2.imread(frame_path)
    h, w = image.shape[:2]

    # Create mask for watermark regions (white = areas to inpaint)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Mark all watermark regions
    for region in regions:
        x = region['x']
        y = region['y']
        width = region['w']
        height = region['h']

        cv2.rectangle(mask,
                      (x, y),
                      (x + width, y + height),
                      255, -1)

    # Convert to PIL Images for LaMa
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask)

    # Apply LaMa inpainting
    result_pil = simple_lama(image_pil, mask_pil)

    # Convert back to OpenCV format
    result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

    # Save result
    cv2.imwrite(output_path, result)

def process_frames(simple_lama, frames_dir, output_dir, regions):
    """Process all frames and remove watermarks using AI"""
    raise_if_cancelled()
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    total_frames = len(frame_files)

    for i, frame_file in enumerate(frame_files):
        raise_if_cancelled()
        frame_path = os.path.join(frames_dir, frame_file)
        output_path = os.path.join(output_dir, frame_file)

        # Remove watermarks
        remove_watermark_lama(simple_lama, frame_path, output_path, regions)

        # Progress feedback - update every frame for real-time progress
        print(f"Processed {i + 1}/{total_frames} frames", file=sys.stderr, flush=True)

        test_sleep = os.environ.get("LAMA_TEST_SLEEP_PER_FRAME")
        if test_sleep:
            with contextlib.suppress(ValueError):
                time.sleep(float(test_sleep))

        raise_if_cancelled()

def encode_video(frames_dir, output_path, fps, ffmpeg_path, original_video):
    """Encode processed frames back to video using FFmpeg with original audio"""
    raise_if_cancelled()
    cmd = [
        ffmpeg_path,
        '-r', str(fps),
        '-i', f'{frames_dir}/frame_%06d.png',
        '-i', original_video,  # Add original video as second input for audio
        '-map', '0:v',  # Use video from first input (processed frames)
        '-map', '1:a',  # Use audio from second input (original video)
        '-map_metadata', '-1',  # Strip all metadata (removes Sora metadata)
        '-c:v', 'libx264',
        '-c:a', 'copy',  # Copy audio without re-encoding
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-shortest',  # Match shortest stream duration
        '-y',
        output_path
    ]
    run_monitored_subprocess(cmd)

def parse_regions(regions_json):
    """Parse watermark regions from JSON string with fallback to defaults"""
    try:
        regions = json.loads(regions_json)
        if not regions:
            print("Warning: No watermark regions provided, using defaults", file=sys.stderr, flush=True)
            # Fallback to default regions for 704x1280 video
            regions = [
                {"x": 280, "y": 65, "w": 170, "h": 90},
                {"x": 277, "y": 1015, "w": 170, "h": 90}
            ]
        return regions
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format for regions: {e}", file=sys.stderr, flush=True)
        raise

def process_single_video(simple_lama, input_video, output_video, ffmpeg_path, regions, video_name=None):
    """
    Process a single video with LaMa AI inpainting.

    Args:
        simple_lama: Loaded LaMa model instance
        input_video: Path to input video file
        output_video: Path to output video file
        ffmpeg_path: Path to FFmpeg executable
        regions: List of watermark regions to remove
        video_name: Optional video name for progress messages (defaults to input filename)

    Raises:
        Exception: If processing fails
    """
    if video_name is None:
        video_name = os.path.basename(input_video)

    fps = 30.0  # Default FPS for Sora videos

    # Create temporary directories for this video
    temp_dir = tempfile.mkdtemp(prefix='lama_video_')
    try:
        frames_dir = os.path.join(temp_dir, 'frames')
        processed_dir = os.path.join(temp_dir, 'processed')
        os.makedirs(frames_dir)
        os.makedirs(processed_dir)

        # Phase 1: Extract frames
        print(f"PHASE|EXTRACTING_FRAMES|{video_name}", file=sys.stderr, flush=True)
        extract_frames(input_video, frames_dir, ffmpeg_path)
        raise_if_cancelled()

        # Phase 2: Process frames with AI
        print(f"PHASE|PROCESSING_FRAMES|{video_name}", file=sys.stderr, flush=True)
        process_frames(simple_lama, frames_dir, processed_dir, regions)
        raise_if_cancelled()

        # Phase 3: Encode video
        print(f"PHASE|ENCODING_VIDEO|{video_name}", file=sys.stderr, flush=True)
        encode_video(processed_dir, output_video, fps, ffmpeg_path, input_video)

    except CancellationError:
        # Best-effort cleanup of any partially written output
        with contextlib.suppress(FileNotFoundError):
            os.remove(output_video)
        raise

    finally:
        # Always clean up temp directory, even if processing failed
        try:
            shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            print(f"Warning: Failed to cleanup temp directory {temp_dir}: {cleanup_error}", file=sys.stderr, flush=True)

def load_model():
    """
    Load LaMa model with bundled model path detection.
    Returns SimpleLama instance ready for processing.
    """
    if os.environ.get("LAMA_SKIP_MODEL_LOAD") == "1":
        print("Test mode: skipping LaMa model load", file=sys.stderr, flush=True)

        class NoOpLama:
            def __call__(self, image, mask):
                return image

        return NoOpLama()

    # Use bundled model if running from PyInstaller bundle
    if getattr(sys, 'frozen', False):
        bundle_dir = sys._MEIPASS
        model_path = os.path.join(bundle_dir, 'big-lama.pt')
        if os.path.exists(model_path):
            os.environ['LAMA_MODEL'] = model_path
            print(f"Using bundled LaMa model (offline mode): {model_path}", file=sys.stderr, flush=True)
        else:
            print("Warning: Bundled model not found, will download from PyTorch Hub (~196MB)", file=sys.stderr, flush=True)
    else:
        print("Running in development mode, will use cached model or download if needed", file=sys.stderr, flush=True)

#    script_dir = os.path.dirname(os.path.abspath(__file__))
#    local_model = os.path.join(script_dir, 'big-lama.pt')
#    if os.path.exists(local_model):
#        os.environ['LAMA_MODEL'] = local_model
#        print(f"Using local LaMa model: {local_model}", file=sys.stderr, flush=True)
#    else:
#        print("Local model not found, will download from PyTorch Hub (~196MB)", file=sys.stderr, flush=True)
#        
        
#    device = "cpu"
#    print("Loading LaMa AI model (CPU mode)...", file=sys.stderr, flush=True)

    # Auto-detect best device for processing
    if torch.backends.mps.is_available():
        device = "mps"
        print("Loading LaMa AI model with GPU acceleration (Metal)...", file=sys.stderr, flush=True)
    elif torch.cuda.is_available():
        device = "cuda"
        print("Loading LaMa AI model with GPU acceleration (CUDA)...", file=sys.stderr, flush=True)
    else:
        device = "cpu"
        print("Loading LaMa AI model (CPU mode - will be slower)...", file=sys.stderr, flush=True)

    print("PHASE|MODEL_LOADING", file=sys.stderr, flush=True)
    simple_lama = SimpleLama(device=device)
    print("PHASE|MODEL_READY", file=sys.stderr, flush=True)

    return simple_lama

def batch_mode(ffmpeg_path):
    """
    Batch processing mode: load model once, process multiple videos from stdin.

    Protocol:
      - Reads from stdin, one line per video: "input_path|output_path|regions_json"
      - Prints to stderr: "COMPLETED|output_path" on success
      - Prints to stderr: "ERROR|input_path|error_message" on failure
      - Continues processing remaining videos even if one fails
      - Exits when stdin is closed
    """
    global MODE
    MODE = "batch"

    # Setup signal handler for cooperative shutdown
    def signal_handler(signum, frame):
        reason = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        request_cancel(reason)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Load model once for all videos
    simple_lama = load_model()

    # Process videos from stdin
    video_count = 0
    success_count = 0
    failure_count = 0

    try:
        for line in sys.stdin:
            if CANCELLED:
                break
            line = line.strip()
            if not line:
                continue

            video_count += 1

            try:
                # Parse input line: "input|output|regions_json"
                parts = line.split('|', 2)
                if len(parts) != 3:
                    raise ValueError(f"Invalid input format: expected 'input|output|regions', got {len(parts)} parts")

                input_video, output_video, regions_json = parts

                # Validate paths
                if not os.path.exists(input_video):
                    raise FileNotFoundError(f"Input video not found: {input_video}")

                # Parse regions
                regions = parse_regions(regions_json)

                # Process video (this will print progress updates)
                video_name = os.path.basename(input_video)
                process_single_video(simple_lama, input_video, output_video, ffmpeg_path, regions, video_name)

                if CANCELLED:
                    break

                # Report success
                print(f"COMPLETED|{output_video}", file=sys.stderr, flush=True)
                success_count += 1

            except CancellationError:
                break
            except Exception as e:
                # Report error but continue processing
                error_msg = str(e).replace('|', ' ')  # Remove pipe chars to avoid protocol confusion
                print(f"ERROR|{parts[0] if len(parts) > 0 else 'unknown'}|{error_msg}", file=sys.stderr, flush=True)
                failure_count += 1

                # Print full traceback for debugging
                traceback.print_exc(file=sys.stderr)

    except CancellationError:
        pass
    except KeyboardInterrupt:
        request_cancel("KeyboardInterrupt")
    except Exception as e:
        print(f"BATCH|FATAL_ERROR|{str(e)}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    finally:
        terminate_active_subprocesses(force=True)

    if CANCELLED:
        return

    # Print final summary
    print(f"BATCH|COMPLETE|total={video_count}|success={success_count}|failed={failure_count}", file=sys.stderr, flush=True)

def single_video_mode():
    """
    Single video processing mode (backward compatible).
    Usage: process_video_lama.py <input_video> <output_video> <ffmpeg_path> <regions_json>
    """
    global MODE
    MODE = "single"

#    if len(sys.argv) != 5:
#        print("Usage: process_video_lama.py <input_video> <output_video> <ffmpeg_path> <regions_json>", file=sys.stderr)
#        print("   or: process_video_lama.py --batch <ffmpeg_path>", file=sys.stderr)
#        sys.exit(1)
#
#    input_video = sys.argv[1]
#    output_video = sys.argv[2]
#    ffmpeg_path = sys.argv[3]
#    regions_json = sys.argv[4]
    if len(sys.argv) != 3:
        print("Usage: process_video_lama.py <input_video> <regions_json>", file=sys.stderr)
        print("  or: process_video_lama.py --batch <ffmpeg_path>", file=sys.stderr)
        # Note: The original batch mode signature is slightly inconsistent with the new
        # single mode, but I kept the batch mode message as you provided it.
        sys.exit(1)

    input_video = sys.argv[1]
    regions_json = sys.argv[2]
    
    # 1. Automatically assign ffmpeg_path
    ffmpeg_path = "./ffmpeg"
    
    # 2. Automatically generate the output_video filename
    # Split the filename into base and extension (e.g., 'video.mp4' -> ('video', '.mp4'))
    base, ext = os.path.splitext(input_video)
    
    # Construct the new filename: base + '-clean' + extension (e.g., 'video-clean.mp4')
    output_video = f"{base}-clean{ext}"


    def signal_handler(signum, frame):
        reason = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        request_cancel(reason)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Parse regions
        regions = parse_regions(regions_json)

        # Load model
        simple_lama = load_model()

        # Process video
        process_single_video(simple_lama, input_video, output_video, ffmpeg_path, regions)

        if CANCELLED:
            return

        print("Done!", file=sys.stderr, flush=True)

    except CancellationError:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        terminate_active_subprocesses(force=True)

def main():
    """
    Main entry point - dispatches to batch or single video mode.
    """
    # Check for batch mode flag
    if len(sys.argv) >= 2 and sys.argv[1] == '--batch':
        if len(sys.argv) != 3:
            print("Usage: process_video_lama.py --batch <ffmpeg_path>", file=sys.stderr)
            sys.exit(1)
        ffmpeg_path = sys.argv[2]
        batch_mode(ffmpeg_path)
    else:
        single_video_mode()

if __name__ == "__main__":
    main()
