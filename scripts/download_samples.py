"""
Autonomous Driving Sample Downloader & Keyframe Extractor.
Ensures reproducible data availability for World Model benchmarks.

Downloads verified open driving video ('car-detection.mp4') from Intel IoT Benchmark
and extracts Frame 56 (vehicle centered in lane) as the canonical benchmark keyframe.
"""

from pathlib import Path
import urllib.request
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw


DRIVING_VIDEO_URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4"


def generate_synthetic_driving_video(output_path: Path, num_frames: int = 75, size: tuple = (256, 256)):
    """Generates a synthetic autonomous driving video (road, lane markings, moving vehicle) if offline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 12.5, size)

    w, h = size
    for t in range(num_frames):
        # Road asphalt (dark gray)
        frame = np.ones((h, w, 3), dtype=np.uint8) * 65

        # Road boundaries and lanes
        cv2.line(frame, (30, 0), (10, h), (180, 180, 180), 2)
        cv2.line(frame, (w - 30, 0), (w - 10, h), (180, 180, 180), 2)

        # Center dashed lane
        lane_dash_offset = (t * 8) % 40
        for y in range(-40 + lane_dash_offset, h, 40):
            cv2.line(frame, (w // 2, y), (w // 2, y + 20), (255, 255, 255), 2)

        # Vehicle (white car moving in lane)
        car_y = int(80 + (t % 30) * 2)
        car_x = w // 2 - 25
        # Car body
        cv2.rectangle(frame, (car_x, car_y), (car_x + 50, car_y + 80), (220, 220, 225), -1)
        cv2.rectangle(frame, (car_x, car_y), (car_x + 50, car_y + 80), (50, 50, 50), 2)
        # Windshield
        cv2.rectangle(frame, (car_x + 6, car_y + 20), (car_x + 44, car_y + 40), (80, 110, 140), -1)
        # Headlights / taillights
        cv2.circle(frame, (car_x + 8, car_y + 75), 4, (40, 40, 220), -1)
        cv2.circle(frame, (car_x + 42, car_y + 75), 4, (40, 40, 220), -1)

        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Generated synthetic driving video at: {output_path}")


def download_or_create_samples(project_root: Path):
    data_raw = project_root / "data" / "raw"
    data_raw.mkdir(parents=True, exist_ok=True)

    img_target = data_raw / "sample_image.jpg"
    vid_target = data_raw / "sample_video.mp4"

    # 1. Video preparation
    if not vid_target.exists():
        temp_mp4 = data_raw / "temp_intel_raw.mp4"
        try:
            print(f"Downloading autonomous driving video from Intel IoT Open Repository...")
            req = urllib.request.Request(DRIVING_VIDEO_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=20) as response, open(temp_mp4, 'wb') as out_file:
                out_file.write(response.read())

            print("Transcoding video clip to standard 256x256 format (75 frames)...")
            cap = cv2.VideoCapture(str(temp_mp4))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(vid_target), fourcc, 12.5, (256, 256))

            count = 0
            while count < 75:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = ImageOps.fit(Image.fromarray(rgb), (256, 256), method=Image.Resampling.LANCZOS)
                out.write(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
                count += 1

            out.release()
            cap.release()
            if temp_mp4.exists():
                temp_mp4.unlink()
            print(f"Standard driving video saved to: {vid_target}")
        except Exception as e:
            print(f"Download video failed ({e}). Generating synthetic driving scene...")
            if temp_mp4.exists():
                temp_mp4.unlink()
            generate_synthetic_driving_video(vid_target)
    else:
        print(f"Driving video already exists: {vid_target}")

    # 2. Extract Canonical Frame 56 (vehicle centered)
    if not img_target.exists():
        print("Extracting canonical Frame 56 (vehicle centered)...")
        cap = cv2.VideoCapture(str(vid_target))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 56)
        ret, frame_bgr = cap.read()
        cap.release()

        if ret:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.save(img_target, quality=95)
            print(f"Canonical Frame 56 saved to: {img_target}")
        else:
            print("Failed to read Frame 56 from video. Generating fallback...")
            img = Image.new("RGB", (256, 256), color=(65, 65, 65))
            draw = ImageDraw.Draw(img)
            draw.rectangle([90, 80, 166, 180], fill=(220, 220, 225), outline=(30, 30, 30), width=2)
            img.save(img_target)
    else:
        print(f"Benchmark image already exists: {img_target}")


if __name__ == "__main__":
    current_root = Path(__file__).resolve().parent.parent
    download_or_create_samples(current_root)
