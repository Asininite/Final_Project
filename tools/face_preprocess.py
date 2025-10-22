"""Face extraction and preprocessing tool

Usage:
    python tools/face_preprocess.py --video-dir data/ffpp/videos --out-dir data/ffpp/aligned --fps 1

This script:
 - extracts frames from videos using ffmpeg (requires ffmpeg installed and on PATH)
 - runs MTCNN face detection and crops faces
 - saves cropped faces to out-dir/<video_name>/frame_00001.png
 - writes a CSV listing (image_path,label,video_name) where label is inferred from path (assumes 'real' or 'fake' in video path or name)

Note: this is a convenience script for local preprocessing on your machine.
"""
import argparse
import os
import subprocess
from pathlib import Path
from PIL import Image
import csv

try:
    from facenet_pytorch import MTCNN
except Exception:
    MTCNN = None


def extract_frames(video_path, out_frames_dir, fps=1):
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    # Use ffmpeg to extract frames
    cmd = [
        'ffmpeg', '-i', str(video_path), '-vf', f'fps={fps}',
        str(out_frames_dir / 'frame_%06d.png')
    ]
    subprocess.run(cmd, check=True)


def detect_and_crop(mtcnn, frame_path, out_dir, max_faces=1, size=224):
    img = Image.open(frame_path).convert('RGB')
    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        return []
    saved = []
    for i, box in enumerate(boxes[:max_faces]):
        left, top, right, bottom = [int(x) for x in box]
        crop = img.crop((left, top, right, bottom)).resize((size, size))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{frame_path.stem}_face{i}.png'
        crop.save(out_path)
        saved.append(out_path)
    return saved


def infer_label_from_path(p: Path):
    # Basic heuristic: check path parts or filename for 'real'/'fake'
    name = p.as_posix().lower()
    if 'real' in name:
        return 0
    if 'fake' in name:
        return 1
    # fallback: unknown
    return -1


def process_videos(video_dir, out_dir, fps=1, size=224, max_faces=1):
    video_dir = Path(video_dir)
    out_dir = Path(out_dir)
    mtcnn = None
    if MTCNN is not None:
        mtcnn = MTCNN(keep_all=True)
    catalog_rows = []

    for video_path in video_dir.glob('**/*'):
        if video_path.suffix.lower() not in ['.mp4', '.mov', '.avi', '.mkv']:
            continue
        video_name = video_path.stem
        frames_dir = out_dir / 'tmp_frames' / video_name
        print('Extracting frames from', video_path)
        extract_frames(video_path, frames_dir, fps=fps)
        for frame_path in frames_dir.glob('*.png'):
            try:
                # Detect and crop
                if mtcnn is None:
                    # If facenet-pytorch not installed, just resize full frame
                    img = Image.open(frame_path).convert('RGB')
                    out_sub = out_dir / video_name
                    out_sub.mkdir(parents=True, exist_ok=True)
                    out_path = out_sub / frame_path.name
                    img.resize((size, size)).save(out_path)
                    saved = [out_path]
                else:
                    saved = detect_and_crop(mtcnn, frame_path, out_dir / video_name, max_faces=max_faces, size=size)
                for p in saved:
                    label = infer_label_from_path(video_path)
                    catalog_rows.append((str(p.relative_to(Path.cwd())), label, video_name))
            except Exception as e:
                print('Error processing', frame_path, e)
        # optional: delete frames_dir to save space
        # shutil.rmtree(frames_dir)
    # write catalog
    csv_path = out_dir / 'catalog.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path','label','video'])
        for r in catalog_rows:
            writer.writerow(r)
    print('Wrote catalog to', csv_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--fps', type=int, default=1)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--max-faces', type=int, default=1)
    args = parser.parse_args()

    if MTCNN is None:
        print('facenet-pytorch not installed. Install with: pip install facenet-pytorch')
    process_videos(args.video_dir, args.out_dir, fps=args.fps, size=args.size, max_faces=args.max_faces)

if __name__ == '__main__':
    main()
