"""
Show a random test image and print ground-truth + generated caption.

Usage:
  python show_random_inference.py [--checkpoint CHECKPOINT]

If no checkpoint given, uses `checkpoints_v2/latest.pth`.
"""
import os
import random
import json
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
import textwrap
from PIL import ImageDraw, ImageFont

from integrated_model import build_model
from caption_generator import Vocabulary


def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    vocab = Vocabulary()
    vocab.word2idx = data.get('word2idx', {})
    idx2word = data.get('idx2word', {})
    try:
        vocab.idx2word = {int(k): v for k, v in idx2word.items()}
    except Exception:
        vocab.idx2word = idx2word
    return vocab


def get_test_captions(captions_file):
    mapping = {}
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            name, cap = line.split(',', 1)
            mapping[name.strip()] = cap.strip()
    return mapping


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    return img, tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (.pth)')
    args = parser.parse_args()

    checkpoint_dir = 'checkpoints_v2'
    default_ckpt = os.path.join(checkpoint_dir, 'latest.pth')
    ckpt_path = args.checkpoint if args.checkpoint else default_ckpt
    if not os.path.exists(ckpt_path):
        print('Checkpoint not found:', ckpt_path)
        return

    vocab_path = os.path.join(checkpoint_dir, 'vocabulary.json')
    if not os.path.exists(vocab_path):
        print('Vocabulary not found:', vocab_path)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    vocab = load_vocab(vocab_path)

    # collect test images
    test_images_dir = os.path.join('data', 'test', 'images')
    imgs = [os.path.join(test_images_dir, p) for p in os.listdir(test_images_dir) if os.path.isfile(os.path.join(test_images_dir, p))]
    if not imgs:
        print('No test images found in', test_images_dir)
        return

    img_path = random.choice(imgs)
    img_name = os.path.basename(img_path)

    # load ground truth caption
    captions_file = os.path.join('data', 'test', 'captions.txt')
    gt_map = get_test_captions(captions_file)
    gt_caption = gt_map.get(img_name, '(no ground-truth caption found)')

    # load model
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get('config', {})
    vocab_size = cfg.get('vocab_size', len(vocab))
    model = build_model(vocab_size, cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # preprocess and infer
    orig_img, tensor = preprocess_image(img_path)
    tensor = tensor.to(device)
    with torch.no_grad():
        outputs = model(images=tensor, captions=None, caption_lengths=None, targets=None, mode='inference')

    caption_ids = outputs['captions'][0]
    generated = vocab.decode(caption_ids.cpu().numpy())

    # DEBUG: print raw token IDs to diagnose
    print('\nDEBUG - Raw token IDs:', caption_ids.cpu().numpy()[:15])  # first 15 tokens
    print('DEBUG - Unique tokens in output:', len(set(caption_ids.cpu().numpy())))
    
    # Print what each token maps to
    token_list = caption_ids.cpu().numpy()[:15]
    print('DEBUG - Token mapping:')
    for i, tid in enumerate(token_list):
        word = vocab.idx2word.get(int(tid), '<UNK>')
        print(f'  Token {tid}: {word}')

    # show image and print captions
    print('\nSelected image:', img_name)
    print('\nGround-truth caption:')
    print('  ', gt_caption)
    print('\nGenerated caption:')
    print('  ', generated)

    # create combined image (original + captions) and show in one window
    try:
        def make_combined(img: Image.Image, gt: str, gen: str, padding=12, bg=(0,0,0)):
            # choose font
            try:
                font = ImageFont.truetype('arial.ttf', size=18)
            except Exception:
                font = ImageFont.load_default()

            # prepare wrapped text lines
            max_width = img.width - padding*2
            def wrap_text(text):
                # estimate approx chars per line
                avg_char_w = max(6, font.getsize('a')[0])
                chars_per_line = max(20, int(max_width / avg_char_w))
                return textwrap.wrap(text, width=chars_per_line)

            gt_lines = wrap_text('GT: ' + gt)
            gen_lines = wrap_text('GEN: ' + gen)

            # compute text height
            draw_tmp = ImageDraw.Draw(img)
            line_h = font.getsize('Ay')[1]
            text_block_h = (len(gt_lines) + len(gen_lines)) * (line_h + 6) + padding*2

            # create new image tall enough
            new_h = img.height + text_block_h
            new_img = Image.new('RGB', (img.width, new_h), color=bg)
            new_img.paste(img, (0, 0))

            draw = ImageDraw.Draw(new_img)
            y = img.height + padding
            # draw a rectangle background for text area
            draw.rectangle([0, img.height, img.width, new_h], fill=(20,20,20))

            for line in gt_lines:
                draw.text((padding, y), line, fill=(255,255,255), font=font)
                y += line_h + 6

            # small separator
            y += 4
            for line in gen_lines:
                draw.text((padding, y), line, fill=(200,240,200), font=font)
                y += line_h + 6

            return new_img

        combined = make_combined(orig_img, gt_caption, generated)
        combined.show()
    except Exception:
        try:
            orig_img.show()
        except Exception:
            pass


if __name__ == '__main__':
    main()
