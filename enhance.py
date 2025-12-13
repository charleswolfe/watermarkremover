import cv2
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def enhance_video_cpu(input_path, output_path, model_name='RealESRGAN_x4plus'):
    # Initialize the model
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4

    # Create upsampler - CPU optimized
    upsampler = RealESRGANer(
        scale=netscale,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=400,  # Use tiling to reduce memory usage
        tile_pad=10,
        pre_pad=0,
        half=False,  # Must be False for CPU
        device='cpu'  # Explicitly set to CPU
    )

    # Open video
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 4, height * 4))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Enhance frame
        output, _ = upsampler.enhance(frame, outscale=4)
        out.write(output)

        frame_count += 1
        print(f"Processed frame {frame_count}/{total_frames} ({frame_count / total_frames * 100:.1f}%)")

    cap.release()
    out.release()
    print("Done!")


# Usage
enhance_video_cpu('input.mp4', 'output_enhanced.mp4')