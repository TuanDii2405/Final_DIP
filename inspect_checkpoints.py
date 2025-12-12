"""
Inspect saved checkpoints: print metadata and run inference on sample test images.

Output per checkpoint is written to `checkpoints_v2/inspections/`:
 - `epoch_XX_results.json` : JSON with caption, num_objects, boxes
 - `epoch_XX_imageNN.jpg` : example image with generated caption overlaid

Usage:
  python inspect_checkpoints.py

You can set environment variable `NUM_SAMPLES` to control how many test images to run (default 3).
"""
import os
import json
import glob
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

from integrated_model import build_model
from caption_generator import Vocabulary


CHECKPOINT_DIR = 'checkpoints_v2'
DATA_TEST_IMAGES = os.path.join('data', 'test', 'images')
INSPECT_DIR = os.path.join(CHECKPOINT_DIR, 'inspections')
os.makedirs(INSPECT_DIR, exist_ok=True)


def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    vocab = Vocabulary()
    vocab.word2idx = data.get('word2idx', {})
    # idx2word keys may be strings
    idx2word = data.get('idx2word', {})
    try:
        vocab.idx2word = {int(k): v for k, v in idx2word.items()}
    except Exception:
        vocab.idx2word = idx2word
    return vocab


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    return img, tensor


def overlay_caption_and_save(image: Image.Image, caption: str, out_path: str):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('arial.ttf', size=18)
    except Exception:
        font = ImageFont.load_default()
    margin = 8
    text = caption
    # compute text size in a compatible way
    try:
        # Pillow >=8: textbbox
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except Exception:
        try:
            w, h = font.getsize(text)
        except Exception:
            # fallback
            w, h = (len(text) * 6, 12)

    rect_h = h + margin * 2
    # draw semi-transparent rectangle (create overlay for alpha if needed)
    try:
        # If image supports alpha, draw with RGBA
        overlay = Image.new('RGBA', (image.width, rect_h), (0, 0, 0, 160))
        image_rgba = image.convert('RGBA')
        image_rgba.paste(overlay, (0, 0), overlay)
        draw = ImageDraw.Draw(image_rgba)
        draw.text((margin, margin), text, fill=(255, 255, 255), font=font)
        # convert back to RGB before saving/showing
        out_image = image_rgba.convert('RGB')
    except Exception:
        # fallback: draw rectangle directly
        draw.rectangle([0, 0, image.width, rect_h], fill=(0, 0, 0))
        draw.text((margin, margin), text, fill=(255, 255, 255), font=font)
        out_image = image

    out_image.save(out_path)
    try:
        out_image.show()
    except Exception:
        pass


def inspect_checkpoint(ckpt_path, vocab, device, sample_images):
    print(f"\n--- Inspecting {ckpt_path} ---")
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = {
        'path': ckpt_path,
        'epoch_in_checkpoint': ckpt.get('epoch'),
        'train_loss': ckpt.get('train_loss'),
        'config': ckpt.get('config')
    }
    print(json.dumps(meta, indent=2, default=str))

    # Build model if possible
    cfg = ckpt.get('config') or {}
    vocab_size = cfg.get('vocab_size', len(vocab))
    try:
        model = build_model(vocab_size, cfg)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Could not build/load model for {ckpt_path}: {e}")
        model = None

    results = []
    for i, img_path in enumerate(sample_images):
        try:
            orig_img, img_tensor = preprocess_image(img_path)
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                outputs = model(images=img_tensor, captions=None, caption_lengths=None, targets=None, mode='inference')

            caption_ids = outputs['captions'][0]
            caption = vocab.decode(caption_ids.cpu().numpy())
            boxes = outputs.get('boxes')
            num_objects = int(outputs.get('num_objects', 0))

            out_img = orig_img.copy()
            out_name = os.path.join(INSPECT_DIR, f"{os.path.basename(ckpt_path).replace('.pth','')}_image{i+1}.jpg")
            overlay_caption_and_save(out_img, caption, out_name)

            result = {
                'image': img_path,
                'caption': caption,
                'num_objects': num_objects,
                'boxes_preview': boxes.cpu().numpy().tolist()[:10] if boxes is not None else []
            }
            results.append(result)
            print(f"Saved annotated image to {out_name}")
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    # Write JSON results
    out_json = os.path.join(INSPECT_DIR, os.path.basename(ckpt_path).replace('.pth', '_results.json'))
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'results': results}, f, indent=2, ensure_ascii=False)
    print(f"Wrote results JSON: {out_json}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    vocab_path = os.path.join(CHECKPOINT_DIR, 'vocabulary.json')
    if not os.path.exists(vocab_path):
        print(f"Vocabulary not found at {vocab_path}")
        return
    vocab = load_vocab(vocab_path)

    # Collect checkpoints epoch_01..epoch_03 (or all epoch_*.pth)
    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'epoch_*.pth')))
    if not ckpts:
        print("No epoch_*.pth checkpoints found in", CHECKPOINT_DIR)
        return

    # sample images: first N from data/test/images
    all_images = sorted(glob.glob(os.path.join(DATA_TEST_IMAGES, '*.*')))
    if not all_images:
        print("No test images found in", DATA_TEST_IMAGES)
        return

    num_samples = int(os.environ.get('NUM_SAMPLES', '3'))
    sample_images = all_images[:num_samples]
    print(f"Will run inference on {len(sample_images)} images: {sample_images}")

    for ckpt in ckpts:
        # only inspect first 3 epochs if present
        base = os.path.basename(ckpt)
        if base in ('epoch_01.pth', 'epoch_02.pth', 'epoch_03.pth'):
            inspect_checkpoint(ckpt, vocab, device, sample_images)


if __name__ == '__main__':
    main()
