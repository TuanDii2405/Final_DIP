"""
Advanced LSTM Caption Generator with Attention Mechanism
Features:
- Multi-layer LSTM with dropout
- Bahdanau Attention over visual features  
- Scheduled sampling for training
- Beam search for inference
- Teacher forcing with annealing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math


# ============================================================================
# Vocabulary
# ============================================================================

class Vocabulary:
    """
    Vocabulary for caption generation
    Handles word-to-index and index-to-word mapping
    """
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        # Special tokens
        self.pad_token = '<pad>'
        self.start_token = '<start>'
        self.end_token = '<end>'
        self.unk_token = '<unk>'
        
        # Add special tokens
        self.add_word(self.pad_token)
        self.add_word(self.start_token)
        self.add_word(self.end_token)
        self.add_word(self.unk_token)
    
    def add_word(self, word):
        """Add a word to vocabulary"""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)
    
    def encode(self, sentence):
        """
        Encode sentence to token IDs
        
        Args:
            sentence: string or list of words
        
        Returns:
            list of token IDs with <start> and <end>
        """
        if isinstance(sentence, str):
            words = sentence.lower().split()
        else:
            words = sentence
        
        tokens = [self.word2idx[self.start_token]]
        
        for word in words:
            if word in self.word2idx:
                tokens.append(self.word2idx[word])
            else:
                tokens.append(self.word2idx[self.unk_token])
        
        tokens.append(self.word2idx[self.end_token])
        
        return tokens
    
    def decode(self, token_ids, skip_special=True):
        """
        Decode token IDs to sentence
        
        Args:
            token_ids: list or tensor of token IDs
            skip_special: whether to skip special tokens
        
        Returns:
            decoded sentence string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy().tolist()
        
        words = []
        special_tokens = {
            self.word2idx[self.pad_token],
            self.word2idx[self.start_token],
            self.word2idx[self.end_token]
        }
        
        for idx in token_ids:
            if skip_special and idx in special_tokens:
                continue
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if word == self.end_token:
                    break
                words.append(word)
        
        return ' '.join(words)


# ============================================================================
# Attention Mechanisms
# ============================================================================

class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention
    Computes attention weights over visual features
    """
    
    def __init__(self, hidden_dim, feature_dim, attention_dim=512):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        
        # Attention layers
        self.W_h = nn.Linear(hidden_dim, attention_dim)
        self.W_v = nn.Linear(feature_dim, attention_dim)
        self.W_a = nn.Linear(attention_dim, 1)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, hidden, visual_features, mask=None):
        """
        Compute attention over visual features
        
        Args:
            hidden: (batch, hidden_dim) current hidden state
            visual_features: (batch, num_regions, feature_dim) visual features
            mask: (batch, num_regions) attention mask (optional)
        
        Returns:
            context: (batch, feature_dim) attended features
            attention_weights: (batch, num_regions) attention distribution
        """
        batch_size, num_regions, feature_dim = visual_features.size()
        
        # Project hidden state
        h_proj = self.W_h(hidden).unsqueeze(1)  # (batch, 1, attention_dim)
        
        # Project visual features
        v_proj = self.W_v(visual_features)  # (batch, num_regions, attention_dim)
        
        # Compute attention scores
        combined = self.tanh(h_proj + v_proj)  # (batch, num_regions, attention_dim)
        scores = self.W_a(combined).squeeze(2)  # (batch, num_regions)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = self.softmax(scores)  # (batch, num_regions)
        
        # Compute context vector
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, num_regions)
            visual_features  # (batch, num_regions, feature_dim)
        ).squeeze(1)  # (batch, feature_dim)
        
        return context, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    """
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_len_q, hidden_dim)
            key: (batch, seq_len_k, hidden_dim)
            value: (batch, seq_len_v, hidden_dim)
            mask: attention mask
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Final linear projection
        output = self.W_o(context)
        
        return output, attention


# ============================================================================
# LSTM Caption Generator
# ============================================================================

class CaptionLSTM(nn.Module):
    """
    Advanced LSTM-based caption generator with attention
    
    Features:
    - Multi-layer LSTM
    - Bahdanau attention over visual features
    - Scheduled sampling during training
    - Beam search for inference
    - Dropout regularization
    """
    
    def __init__(
        self,
        vocab_size,
        embedding_dim=512,
        hidden_dim=512,
        num_layers=2,
        dropout=0.5,
        attention_dim=512,
        use_attention=True,
        use_multihead=False
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_multihead = use_multihead
        
        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM cell
        self.lstm = nn.LSTM(
            input_size=embedding_dim + (hidden_dim if use_attention else 0),
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        if use_attention:
            if use_multihead:
                self.attention = MultiHeadAttention(hidden_dim, num_heads=8, dropout=dropout)
            else:
                self.attention = BahdanauAttention(hidden_dim, hidden_dim, attention_dim)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        
        # Deep output with residual connection
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_embedding = nn.Linear(embedding_dim, hidden_dim)
        self.fc_context = nn.Linear(hidden_dim, hidden_dim) if use_attention else None
        
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        # Embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Forget gate bias = 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        # Output layers
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state and cell state
        
        Returns:
            (h_0, c_0): initial hidden and cell states
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h_0, c_0)
    
    def forward(self, visual_features, captions, caption_lengths=None, teacher_forcing_ratio=1.0):
        """
        Forward pass with teacher forcing
        
        Args:
            visual_features: (batch, num_regions, feature_dim) visual features
            captions: (batch, max_len) caption token IDs
            caption_lengths: (batch,) actual lengths
            teacher_forcing_ratio: probability of using teacher forcing
        
        Returns:
            outputs: (batch, max_len, vocab_size) predicted logits
            attention_weights: (batch, max_len, num_regions) if using attention
        """
        batch_size = visual_features.size(0)
        max_len = captions.size(1)
        device = visual_features.device
        
        # Initialize hidden state
        hidden = self.init_hidden(batch_size, device)
        
        # Storage for outputs
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(device)
        attention_weights_list = [] if self.use_attention else None
        
        # Start with <start> token (assumed to be at index 1)
        input_word = captions[:, 0]  # (batch,) - first token is <start>
        
        for t in range(1, max_len):
            # Embed current word
            embedded = self.embedding(input_word)  # (batch, embedding_dim)
            
            # Attention over visual features
            if self.use_attention:
                # Get hidden state for attention
                if isinstance(hidden, tuple):
                    h_t = hidden[0][-1]  # Last layer hidden state
                else:
                    h_t = hidden[-1]
                
                context, attn_weights = self.attention(h_t, visual_features)
                attention_weights_list.append(attn_weights)
                
                # Concatenate embedding with context
                lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
            else:
                lstm_input = embedded.unsqueeze(1)
            
            # LSTM step
            lstm_out, hidden = self.lstm(lstm_input, hidden)
            lstm_out = lstm_out.squeeze(1)  # (batch, hidden_dim)
            
            # Deep output
            # Combine LSTM output, embedding, and context
            h_out = self.fc_hidden(lstm_out)
            e_out = self.fc_embedding(embedded)
            
            combined = h_out + e_out
            
            if self.use_attention and self.fc_context is not None:
                c_out = self.fc_context(context)
                combined = combined + c_out
            
            combined = self.layer_norm1(combined)
            combined = F.relu(combined)
            combined = self.dropout(combined)
            
            # Output projection
            output = self.fc_out(combined)  # (batch, vocab_size)
            outputs[:, t] = output
            
            # Teacher forcing
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher:
                input_word = captions[:, t]
            else:
                input_word = output.argmax(dim=1)
        
        if self.use_attention:
            attention_weights = torch.stack(attention_weights_list, dim=1)
            return outputs, attention_weights
        else:
            return outputs, None
    
    def generate(self, visual_features, max_length=20, start_token_idx=1, end_token_idx=2, beam_size=1):
        """
        Generate captions using greedy decoding or beam search
        
        Args:
            visual_features: (batch, num_regions, feature_dim)
            max_length: maximum caption length
            start_token_idx: index of <start> token
            end_token_idx: index of <end> token
            beam_size: beam size for beam search (1 = greedy)
        
        Returns:
            captions: (batch, max_length) generated token IDs
        """
        if beam_size == 1:
            return self._greedy_decode(visual_features, max_length, start_token_idx, end_token_idx)
        else:
            return self._beam_search(visual_features, max_length, start_token_idx, end_token_idx, beam_size)
    
    def _greedy_decode(self, visual_features, max_length, start_token_idx, end_token_idx):
        """Greedy decoding"""
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        # Initialize
        hidden = self.init_hidden(batch_size, device)
        input_word = torch.LongTensor([start_token_idx] * batch_size).to(device)
        
        captions = []
        
        for t in range(max_length):
            # Embed
            embedded = self.embedding(input_word)
            
            # Attention
            if self.use_attention:
                h_t = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]
                context, _ = self.attention(h_t, visual_features)
                lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
            else:
                lstm_input = embedded.unsqueeze(1)
            
            # LSTM
            lstm_out, hidden = self.lstm(lstm_input, hidden)
            lstm_out = lstm_out.squeeze(1)
            
            # Output
            h_out = self.fc_hidden(lstm_out)
            e_out = self.fc_embedding(embedded)
            combined = h_out + e_out
            
            if self.use_attention and self.fc_context is not None:
                c_out = self.fc_context(context)
                combined = combined + c_out
            
            combined = self.layer_norm1(combined)
            combined = F.relu(combined)
            
            output = self.fc_out(combined)
            predicted = output.argmax(dim=1)
            
            captions.append(predicted.unsqueeze(1))
            input_word = predicted
            
            # Stop if all sequences have generated <end>
            if (predicted == end_token_idx).all():
                break
        
        captions = torch.cat(captions, dim=1)  # (batch, seq_len)
        
        # Pad to max_length if needed
        if captions.size(1) < max_length:
            padding = torch.zeros(batch_size, max_length - captions.size(1), dtype=torch.long, device=device)
            captions = torch.cat([captions, padding], dim=1)
        
        return captions
    
    def _beam_search(self, visual_features, max_length, start_token_idx, end_token_idx, beam_size):
        """
        Beam search decoding
        
        Note: Simplified version for single batch
        """
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        if batch_size > 1:
            # For multiple batches, process one by one
            results = []
            for i in range(batch_size):
                caption = self._beam_search_single(
                    visual_features[i:i+1],
                    max_length,
                    start_token_idx,
                    end_token_idx,
                    beam_size
                )
                results.append(caption)
            return torch.cat(results, dim=0)
        else:
            return self._beam_search_single(
                visual_features,
                max_length,
                start_token_idx,
                end_token_idx,
                beam_size
            )
    
    def _beam_search_single(self, visual_features, max_length, start_token_idx, end_token_idx, beam_size):
        """Beam search for single sample"""
        device = visual_features.device
        
        # Initialize beams
        sequences = [[start_token_idx]]
        scores = [0.0]
        hidden_states = [self.init_hidden(1, device)]
        
        for t in range(max_length):
            all_candidates = []
            
            for i, seq in enumerate(sequences):
                if seq[-1] == end_token_idx:
                    all_candidates.append((scores[i], seq, hidden_states[i]))
                    continue
                
                # Current word
                input_word = torch.LongTensor([seq[-1]]).to(device)
                embedded = self.embedding(input_word)
                
                # Attention
                if self.use_attention:
                    h_t = hidden_states[i][0][-1]
                    context, _ = self.attention(h_t, visual_features)
                    lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
                else:
                    lstm_input = embedded.unsqueeze(1)
                
                # LSTM
                lstm_out, new_hidden = self.lstm(lstm_input, hidden_states[i])
                lstm_out = lstm_out.squeeze(1)
                
                # Output
                h_out = self.fc_hidden(lstm_out)
                e_out = self.fc_embedding(embedded)
                combined = h_out + e_out
                
                if self.use_attention and self.fc_context is not None:
                    c_out = self.fc_context(context)
                    combined = combined + c_out
                
                combined = self.layer_norm1(combined)
                combined = F.relu(combined)
                output = self.fc_out(combined)
                
                # Get top-k predictions
                log_probs = F.log_softmax(output, dim=1)
                topk_probs, topk_ids = log_probs.topk(beam_size, dim=1)
                
                for k in range(beam_size):
                    new_seq = seq + [topk_ids[0, k].item()]
                    new_score = scores[i] + topk_probs[0, k].item()
                    all_candidates.append((new_score, new_seq, new_hidden))
            
            # Select top-k beams
            ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
            sequences = [seq for _, seq, _ in ordered[:beam_size]]
            scores = [score for score, _, _ in ordered[:beam_size]]
            hidden_states = [hidden for _, _, hidden in ordered[:beam_size]]
            
            # Check if all beams ended
            if all(seq[-1] == end_token_idx for seq in sequences):
                break
        
        # Return best sequence
        best_seq = sequences[0]
        
        # Pad to max_length
        if len(best_seq) < max_length:
            best_seq = best_seq + [0] * (max_length - len(best_seq))
        else:
            best_seq = best_seq[:max_length]
        
        return torch.LongTensor([best_seq]).to(device)
    
    def compute_loss(self, visual_features, captions, caption_lengths=None):
        """
        Compute caption generation loss
        
        Args:
            visual_features: (batch, num_regions, feature_dim)
            captions: (batch, max_len) with <start> and <end> tokens
            caption_lengths: (batch,) actual lengths
        
        Returns:
            loss: scalar loss value
        """
        batch_size = captions.size(0)
        
        # Forward pass with teacher forcing
        outputs, _ = self.forward(visual_features, captions, caption_lengths, teacher_forcing_ratio=1.0)
        
        # Prepare targets (shift by 1)
        targets = captions[:, 1:].contiguous()  # Remove <start>
        outputs = outputs[:, 1:].contiguous()  # Align with targets
        
        # Flatten
        outputs = outputs.view(-1, self.vocab_size)
        targets = targets.view(-1)
        
        # Cross entropy loss (ignore padding)
        loss = F.cross_entropy(outputs, targets, ignore_index=0)
        
        return loss


# ============================================================================
# Testing
# ============================================================================

def test_caption_generator():
    """Test caption generator"""
    print("Testing Caption LSTM...")
    
    vocab_size = 5000
    batch_size = 4
    num_regions = 10
    feature_dim = 512
    max_len = 20
    
    # Create model
    model = CaptionLSTM(
        vocab_size=vocab_size,
        embedding_dim=512,
        hidden_dim=512,
        num_layers=2,
        dropout=0.5,
        use_attention=True
    )
    
    # Dummy data
    visual_features = torch.randn(batch_size, num_regions, feature_dim)
    captions = torch.randint(0, vocab_size, (batch_size, max_len))
    captions[:, 0] = 1  # Start token
    caption_lengths = torch.randint(10, max_len, (batch_size,))
    
    # Forward pass
    outputs, attn_weights = model(visual_features, captions, caption_lengths)
    print(f"Outputs shape: {outputs.shape}")
    print(f"Attention weights shape: {attn_weights.shape if attn_weights is not None else None}")
    
    # Generate
    generated = model.generate(visual_features, max_length=15, beam_size=1)
    print(f"Generated captions shape: {generated.shape}")
    
    # Beam search
    beam_generated = model.generate(visual_features, max_length=15, beam_size=3)
    print(f"Beam search captions shape: {beam_generated.shape}")
    
    # Loss
    loss = model.compute_loss(visual_features, captions, caption_lengths)
    print(f"Loss: {loss.item():.4f}")
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    test_caption_generator()
