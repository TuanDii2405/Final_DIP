"""
Test trained model - Generate captions and detect objects
"""
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import json
import sys

from integrated_model import build_model
from caption_generator import Vocabulary


def load_model(checkpoint_path, vocab_size, device):
    """Load trained model from checkpoint"""
    config = {
        'num_classes': 91,
        'vocab_size': vocab_size,
        'embed_dim': 512,
        'embedding_dim': 512,
        'hidden_dim': 512,
        'num_relationship_layers': 3,
        'num_spatial_rels': 8,
        'num_semantic_rels': 50,
    }
    
    model = build_model(vocab_size, config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    
    return model


def load_vocabulary(vocab_path):
    """Load vocabulary"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    vocab = Vocabulary()
    vocab.word2idx = vocab_data['word2idx']
    vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
    
    print(f"✓ Loaded vocabulary: {len(vocab)} words")
    return vocab


def preprocess_image(image_path):
    """Preprocess image for model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    
    return image, image_tensor


def test_image(model, vocabulary, image_path, device):
    """Test model on a single image"""
    print(f"\n{'='*70}")
    print(f"Testing image: {image_path}")
    print(f"{'='*70}")
    
    # Load and preprocess image
    original_image, image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Run model inference
    with torch.no_grad():
        outputs = model(
            images=image_tensor,
            captions=None,
            caption_lengths=None,
            targets=None,
            mode='inference'
        )
    
    # 1. GENERATED CAPTION
    caption_ids = outputs['captions'][0]  # (seq_len,)
    caption = vocabulary.decode(caption_ids.cpu().numpy())
    
    print(f"\n📝 GENERATED CAPTION:")
    print(f"   '{caption}'")
    
    # 2. DETECTED OBJECTS
    boxes = outputs['boxes']
    num_objects = outputs['num_objects']
    
    print(f"\n🎯 DETECTED OBJECTS: {num_objects} objects")
    
    if num_objects > 0:
        print(f"\n   Object Boxes (x1, y1, x2, y2):")
        for i, box in enumerate(boxes[:min(10, num_objects)]):
            x1, y1, x2, y2 = box.cpu().numpy()
            print(f"   [{i+1}] Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            
        # Get image size for percentage
        w, h = original_image.size
        print(f"\n   (Image size: {w}x{h})")
    else:
        print(f"   No objects detected with high confidence")
    
    print(f"\n{'='*70}\n")
    
    return {
        'caption': caption,
        'num_objects': num_objects,
        'boxes': boxes.cpu().numpy() if num_objects > 0 else []
    }


def main():
    """Main test function"""
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Paths
    checkpoint_path = 'checkpoints_v2/epoch_1.pth'  # Change to your checkpoint
    vocab_path = 'checkpoints_v2/vocabulary.json'
    
    # Test images
    test_images = [
        'data/test/images/1000268201_693b08cb0e.jpg',  # Example paths
        'data/test/images/1001773457_577c3a7d70.jpg',
        'data/test/images/1002674143_1b742ab4b8.jpg',
    ]
    
    # Load model and vocabulary
    print("Loading model...")
    vocabulary = load_vocabulary(vocab_path)
    model = load_model(checkpoint_path, len(vocabulary), device)
    
    # Test each image
    results = []
    for img_path in test_images:
        try:
            result = test_image(model, vocabulary, img_path, device)
            results.append(result)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for i, (img_path, result) in enumerate(zip(test_images, results)):
        print(f"\n{i+1}. {img_path.split('/')[-1]}")
        print(f"   Caption: '{result['caption']}'")
        print(f"   Objects: {result['num_objects']}")
    
    print("\n✓ Testing complete!")


if __name__ == '__main__':
    main()
