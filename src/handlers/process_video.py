from src.services.process_video_lama import load_model, process_single_video
from typing import Callable, Optional


def handle_process_video(
        input_path: str,
        output_path: str,
        use_cpu: bool,
        upscale: bool,
        rectangles: list,
        progress_cb: Optional[Callable[[int], None]] = None
                  ):
    simple_lama = load_model()

    process_single_video(
        simple_lama,
        input_path,
        output_path,
        rectangles,
        progress_cb=progress_cb
    )


    # Process with your Real-ESRGAN code
    # ... your processing logic ...