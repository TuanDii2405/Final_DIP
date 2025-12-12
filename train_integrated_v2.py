"""
Training Script for Integrated Caption Model
Train on reorganized Flickr8k dataset with proper train/test split
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
import argparse

# Import model components
from integrated_model import build_model, CombinedLoss
from caption_generator import Vocabulary


# ============================================================================
# Dataset
# ============================================================================

class Flickr8kDataset(Dataset):
    """
    Flickr8k dataset loader
    Reads from reorganized data/train or data/test folders
    """
    
    def __init__(self, data_dir, transform=None, max_caption_len=20):
        """
        Args:
            data_dir: path to data/train or data/test
            transform: image transformations
            max_caption_len: maximum caption length
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.caption_file = os.path.join(data_dir, 'captions.txt')
        self.transform = transform
        self.max_caption_len = max_caption_len
        
        # Load captions
        self.image_caption_pairs = []
        with open(self.caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or ',' not in line:
                    continue
                img_name, caption = line.split(',', 1)
                img_path = os.path.join(self.image_dir, img_name)
                if os.path.exists(img_path):
                    self.image_caption_pairs.append((img_path, caption.strip()))
        
        print(f"Loaded {len(self.image_caption_pairs)} image-caption pairs from {data_dir}")
    
    def __len__(self):
        return len(self.image_caption_pairs)
    
    def __getitem__(self, idx):
        img_path, caption = self.image_caption_pairs[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Fallback to first image if error
            image = Image.open(self.image_caption_pairs[0][0]).convert('RGB')
            caption = self.image_caption_pairs[0][1]
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'caption': caption,
            'image_path': img_path
        }


def collate_fn(batch, vocabulary, max_len=20):
    """
    Custom collate function to handle variable length captions
    """
    images = torch.stack([item['image'] for item in batch])
    captions_text = [item['caption'] for item in batch]
    
    # Encode captions
    captions_encoded = []
    caption_lengths = []
    
    for caption in captions_text:
        tokens = vocabulary.encode(caption)
        # Pad or truncate
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [vocabulary.word2idx['<pad>']] * (max_len - len(tokens))
        
        captions_encoded.append(tokens)
        caption_lengths.append(min(len(vocabulary.encode(caption)), max_len))
    
    captions = torch.LongTensor(captions_encoded)
    caption_lengths = torch.LongTensor(caption_lengths)
    
    return {
        'images': images,
        'captions': captions,
        'caption_lengths': caption_lengths,
        'captions_text': captions_text
    }


# ============================================================================
# Build Vocabulary from Training Data
# ============================================================================

def build_vocabulary(caption_file, min_word_freq=5, max_vocab_size=10000):
    """
    Build vocabulary from training captions
    """
    print("Building vocabulary...")
    
    word_freq = {}
    
    with open(caption_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            _, caption = line.split(',', 1)
            words = caption.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
    
    # Filter by frequency
    words = [w for w, freq in word_freq.items() if freq >= min_word_freq]
    words = sorted(words, key=lambda w: word_freq[w], reverse=True)
    
    # Limit vocabulary size
    words = words[:max_vocab_size - 4]  # Reserve for special tokens
    
    # Create vocabulary
    vocab = Vocabulary()
    for word in words:
        vocab.add_word(word)
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Most common words: {words[:20]}")
    
    return vocab


# ============================================================================
# Training Function
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, vocabulary, device, epoch):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    caption_loss_sum = 0
    detection_loss_sum = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        captions = batch['captions'].to(device)
        caption_lengths = batch['caption_lengths'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Note: targets for Faster R-CNN not available in caption-only dataset
        # Model will train primarily on caption loss
        outputs = model(
            images,
            captions=captions,
            caption_lengths=caption_lengths,
            targets=None,
            mode='train'
        )
        
        # Compute loss
        loss = criterion(outputs)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Logging
        total_loss += loss.item()
        caption_loss_sum += outputs.get('caption_loss', 0)
        
        det_loss = sum([
            outputs.get('loss_objectness', 0),
            outputs.get('loss_rpn_box_reg', 0),
            outputs.get('loss_classifier', 0),
            outputs.get('loss_box_reg', 0)
        ])
        if isinstance(det_loss, torch.Tensor):
            det_loss = det_loss.item()
        detection_loss_sum += det_loss
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cap': f'{outputs.get("caption_loss", 0):.4f}',
            'det': f'{det_loss:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_caption_loss = caption_loss_sum / len(dataloader)
    avg_det_loss = detection_loss_sum / len(dataloader)
    
    return avg_loss, avg_caption_loss, avg_det_loss


def evaluate(model, dataloader, criterion, vocabulary, device):
    """
    Evaluate model on validation set
    """
    model.eval()
    total_loss = 0
    caption_loss_sum = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['images'].to(device)
            captions = batch['captions'].to(device)
            caption_lengths = batch['caption_lengths'].to(device)
            
            outputs = model(
                images,
                captions=captions,
                caption_lengths=caption_lengths,
                targets=None,
                mode='train'
            )
            
            loss = criterion(outputs)
            total_loss += loss.item()
            caption_loss_sum += outputs.get('caption_loss', 0)
    
    avg_loss = total_loss / len(dataloader)
    avg_caption_loss = caption_loss_sum / len(dataloader)
    
    return avg_loss, avg_caption_loss


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    """
    Main training function
    """
    # Configuration
    config = {
        'num_classes': 91,
        'vocab_size': 10000,
        'embed_dim': 512,  # Changed from embedding_dim to embed_dim
        'embedding_dim': 512,  # Keep both for compatibility
        'hidden_dim': 512,
        'num_relationship_layers': 3,
        'num_spatial_rels': 8,
        'num_semantic_rels': 50,
        'pretrained_backbone': True,
        'batch_size': 1,  # Reduced for debugging
        'num_epochs': 20,
        'learning_rate': 1e-5,  # Reduced from 1e-4 for fine-tuning
        'weight_decay': 1e-5,
        'max_caption_len': 20,
        'checkpoint_dir': 'checkpoints_v2',
        'log_file': 'training_v2.log'
    }
    # Parse CLI args
    parser = argparse.ArgumentParser(description='Train integrated model (supports resume).')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (default: latest in checkpoint_dir)')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Build vocabulary
    vocabulary = build_vocabulary(
        'data/train/captions.txt',
        min_word_freq=5,
        max_vocab_size=config['vocab_size']
    )
    
    # Save vocabulary
    vocab_path = os.path.join(config['checkpoint_dir'], 'vocabulary.json')
    with open(vocab_path, 'w') as f:
        json.dump({
            'word2idx': vocabulary.word2idx,
            'idx2word': vocabulary.idx2word
        }, f)
    print(f"Vocabulary saved to {vocab_path}")
    
    # Update config with actual vocab size
    config['vocab_size'] = len(vocabulary)
    
    # Image transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = Flickr8kDataset(
        'data/train',
        transform=train_transform,
        max_caption_len=config['max_caption_len']
    )
    
    test_dataset = Flickr8kDataset(
        'data/test',
        transform=test_transform,
        max_caption_len=config['max_caption_len']
    )
    
    # Optional: Use only 80% of train data for faster training
    # Uncomment these lines to use subset:
    # train_size = int(0.8 * len(train_dataset))
    # train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
    # print(f"Using {len(train_dataset)} samples for training (80% of original)")
    
    # Dataloaders (num_workers=0 for Windows compatibility)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, vocabulary, config['max_caption_len'])
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, vocabulary, config['max_caption_len'])
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(len(vocabulary), config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    best_loss = float('inf')
    training_history = []
    
    # If resuming, append to log, otherwise overwrite
    log_mode = 'a' if args.resume else 'w'
    log_file = open(config['log_file'], log_mode)
    log_file.write(f"Training started at {datetime.now()}\n")
    log_file.write(f"Configuration: {json.dumps(config, indent=2)}\n\n")

    # Path for training history
    history_path = os.path.join(config['checkpoint_dir'], 'training_history.json')

    # Determine resume checkpoint: explicit --resume has priority, otherwise try latest in checkpoint_dir
    resume_path = None
    if args.resume:
        resume_path = args.resume
    else:
        candidate = os.path.join(config['checkpoint_dir'], 'latest.pth')
        if os.path.exists(candidate):
            resume_path = candidate

    start_epoch = 1
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        # Load model + optimizer + scheduler states
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Warning: could not load full model state: {e}")
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"Warning: could not load optimizer state: {e}")
        try:
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            print(f"Warning: could not load scheduler state: {e}")

        # Move optimizer state tensors to the correct device (important when resuming on different device)
        for state in optimizer.state.values():
            for k, v in list(state.items()):
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = checkpoint.get('epoch', 0) + 1
        best_loss = checkpoint.get('train_loss', best_loss)

        # Try to load training history if present
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as hf:
                    training_history = json.load(hf)
            except Exception as e:
                print(f"Warning: could not load training history: {e}")
    
    for epoch in range(start_epoch, config['num_epochs'] + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Train
        train_loss, train_cap_loss, train_det_loss = train_epoch(
            model, train_loader, optimizer, criterion, vocabulary, device, epoch
        )
        
        # Learning rate scheduling (based on train loss instead)
        scheduler.step(train_loss)
        
        epoch_time = time.time() - start_time
        
        # Logging (only train loss)
        log_msg = (
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} (Cap: {train_cap_loss:.4f}, Det: {train_det_loss:.4f}) | "
            f"Time: {epoch_time:.1f}s | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        print(f"\n{log_msg}")
        log_file.write(log_msg + '\n')
        log_file.flush()
        
        # Save history (without test loss)
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_caption_loss': train_cap_loss,
            'train_detection_loss': train_det_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'time': epoch_time
        })
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'train_loss': train_loss
        }
        
        # Save latest
        latest_path = os.path.join(config['checkpoint_dir'], 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save checkpoint after EVERY epoch
        epoch_path = os.path.join(config['checkpoint_dir'], f'epoch_{epoch:02d}.pth')
        torch.save(checkpoint, epoch_path)
        print(f"✓ Checkpoint saved: {epoch_path}")
    
    # Training complete - now run test evaluation ONCE
    print("\n" + "="*80)
    print("TRAINING COMPLETE! Running final test evaluation...")
    print("="*80)
    
    # Final test evaluation
    test_loss, test_cap_loss = evaluate(
        model, test_loader, criterion, vocabulary, device
    )
    
    print(f"\n{'='*80}")
    print(f"FINAL TEST RESULTS:")
    print(f"Test Loss: {test_loss:.4f} (Caption Loss: {test_cap_loss:.4f})")
    print(f"{'='*80}")
    
    # Save final model with test results
    final_checkpoint = {
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'test_caption_loss': test_cap_loss
    }
    final_path = os.path.join(config['checkpoint_dir'], 'final_model.pth')
    torch.save(final_checkpoint, final_path)
    print(f"\nFinal model saved: {final_path}")
    
    # Training complete
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print(f"Final test loss: {test_loss:.4f}")
    
    # Save training history
    history_path = os.path.join(config['checkpoint_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    log_file.write(f"\nTraining completed at {datetime.now()}\n")
    log_file.write(f"Final test loss: {test_loss:.4f}\n")
    log_file.write(f"Final test caption loss: {test_cap_loss:.4f}\n")
    log_file.close()
    
    print(f"\nLogs saved to: {config['log_file']}")
    print(f"History saved to: {history_path}")


if __name__ == '__main__':
    main()
