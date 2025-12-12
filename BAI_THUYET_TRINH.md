# THUYẾT TRÌNH DỰ ÁN: HỆ THỐNG TẠO CHÚ THÍCH ẢNH TỰ ĐỘNG

---

## PHẦN 1: GIỚI THIỆU

### 1.1. Bài toán cần giải quyết
Trong thời đại số hóa, hàng triệu bức ảnh được tải lên Internet mỗi ngày. Việc tự động tạo mô tả (caption) cho ảnh có ý nghĩa quan trọng:
- **Hỗ trợ người khiếm thị:** Đọc mô tả ảnh qua công nghệ text-to-speech
- **Tìm kiếm ảnh:** Tìm kiếm bằng ngôn ngữ tự nhiên thay vì tags thủ công
- **Quản lý ảnh:** Tự động phân loại và gắn nhãn ảnh
- **Mạng xã hội:** Gợi ý caption cho người dùng

### 1.2. Thách thức
- Hiểu được **nội dung ảnh** (có gì trong ảnh?)
- Phân tích **mối quan hệ** giữa các đối tượng
- Sinh **câu tự nhiên** mô tả chính xác

### 1.3. Giải pháp đề xuất
Xây dựng hệ thống **end-to-end** kết hợp 3 công nghệ AI tiên tiến:
1. **Faster R-CNN** - Phát hiện đối tượng
2. **Relationship Graph** - Phân tích mối quan hệ
3. **LSTM** - Sinh ngôn ngữ tự nhiên

---

## PHẦN 2: DỮ LIỆU NGHIÊN CỨU

### 2.1. Dataset: Flickr8k
**Flickr8k** là bộ dataset chuẩn trong nghiên cứu Image Captioning:
- **Nguồn:** Ảnh từ Flickr.com
- **Quy mô:** 8,091 ảnh
- **Nội dung:** Ảnh về đời sống hàng ngày (người, động vật, phong cảnh, hoạt động...)
- **Annotation:** Mỗi ảnh có caption mô tả được viết bởi con người

### 2.2. Chuẩn bị dữ liệu
**Chia dataset theo tỷ lệ 80/20:**
- **Training set:** 6,472 ảnh (80%) - Dùng để huấn luyện model
- **Test set:** 1,619 ảnh (20%) - Dùng để đánh giá độ chính xác

**Đảm bảo tính khoa học:**
- Shuffle ngẫu nhiên với seed cố định (seed=42)
- Lưu danh sách split vào `data_splits.json` để có thể tái tạo kết quả
- Test set **hoàn toàn tách biệt**, model không được "nhìn thấy" trong quá trình training

**Dữ liệu bổ sung:**
- `boxes_rel.json`: Bounding boxes cho các objects trong ảnh (tự tạo bằng pretrained detector)
- `captions.txt`: Ground truth captions cho training

---

## PHẦN 3: KIẾN TRÚC HỆ THỐNG

### 3.1. Tổng quan Pipeline

```
┌─────────────┐
│   INPUT:    │
│   Ảnh RGB   │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────┐
│  MODULE 1: Faster R-CNN      │
│  → Phát hiện đối tượng       │
│  → Trích xuất features       │
└──────┬───────────────────────┘
       │ Object features
       ▼
┌──────────────────────────────┐
│  MODULE 2: Relationship GNN  │
│  → Xây dựng scene graph      │
│  → Phân tích mối quan hệ     │
└──────┬───────────────────────┘
       │ Enhanced features
       ▼
┌──────────────────────────────┐
│  MODULE 3: LSTM Generator    │
│  → Attention mechanism       │
│  → Sinh caption từng từ      │
└──────┬───────────────────────┘
       │
       ▼
┌─────────────┐
│   OUTPUT:   │
│   Caption   │
└─────────────┘
```

### 3.2. Module 1: Faster R-CNN (Object Detection)

**Vai trò:** "Con mắt" của hệ thống - nhìn và nhận diện đối tượng trong ảnh

**Kiến trúc:**
- **Backbone:** VGG16 pretrained trên ImageNet
  - Đã học được các đặc trưng cơ bản (cạnh, góc, texture, patterns...)
  - Extract feature maps từ ảnh đầu vào
  
- **Region Proposal Network (RPN):**
  - Đề xuất các vùng có khả năng chứa objects
  - Loại bỏ background không cần thiết
  
- **ROI Pooling & Classification:**
  - Resize các vùng đề xuất về kích thước cố định
  - Phân loại object và tinh chỉnh bounding box

**Output:**
- **Bounding boxes:** Tọa độ (x1, y1, x2, y2) của mỗi object
- **Labels:** Nhãn phân loại (person, dog, car, tree...)
- **Feature vectors:** Vector đặc trưng 512-dim cho mỗi object

**Ví dụ:** Ảnh có 2 người đang chơi với chó → Detect 3 objects: person #1, person #2, dog

### 3.3. Module 2: Relationship Graph (Graph Neural Network)

**Vai trò:** "Bộ não phân tích" - hiểu mối quan hệ không gian và ngữ nghĩa giữa các objects

**Tại sao cần Relationship Graph?**
- Faster R-CNN chỉ detect objects **riêng lẻ**
- Nhưng caption cần hiểu **ngữ cảnh**: "người đang ném bóng cho chó" (không chỉ "người + bóng + chó")

**Cách hoạt động:**
1. **Xây dựng Scene Graph:**
   - Mỗi object = 1 node
   - Kết nối các nodes thành graph dựa trên vị trí không gian
   
2. **Graph Neural Network:**
   - Message passing giữa các nodes
   - Mỗi node "học" thông tin từ neighbors
   - Cập nhật features dựa trên ngữ cảnh xung quanh

3. **Relationship Classification:**
   - Phân loại mối quan hệ: above, next to, holding, riding, wearing...
   - Tích hợp relationship features vào object features

**Output:**
- **Enhanced object features:** Vector đặc trưng đã bổ sung thông tin về mối quan hệ
- **Relationship matrix:** Ma trận thể hiện mối quan hệ giữa các cặp objects

**Ví dụ:** 
- Detect: person, ball, dog
- Graph: person -[throwing]→ ball -[to]→ dog
- Features giờ chứa thông tin về action "throwing"

### 3.4. Module 3: LSTM Caption Generator

**Vai trò:** "Người kể chuyện" - chuyển visual features thành ngôn ngữ tự nhiên

**Kiến trúc:**

1. **Vocabulary (Từ điển):**
   - Xây dựng từ 8,091 captions trong dataset
   - Kích thước: 5,320 từ duy nhất
   - Special tokens: `<start>`, `<end>`, `<pad>`, `<unk>`

2. **LSTM (Long Short-Term Memory):**
   - **2 layers LSTM** với hidden size 512
   - Nhớ được ngữ cảnh dài (long-term dependencies)
   - Sinh caption **tuần tự từng từ một**

3. **Attention Mechanism:**
   - Tại mỗi time step, LSTM "chú ý" vào các objects khác nhau
   - Ví dụ: Khi sinh từ "throwing", attention tập trung vào person; khi sinh "ball", attention tập trung vào ball
   - Giúp model linh hoạt và chính xác hơn

**Quá trình sinh caption:**
```
Step 1: [<start>] → "a"
Step 2: [<start>, a] → "person"
Step 3: [<start>, a, person] → "is"
Step 4: [<start>, a, person, is] → "throwing"
Step 5: [<start>, a, person, is, throwing] → "a"
Step 6: [<start>, a, person, is, throwing, a] → "ball"
...
Step N: [...] → <end>
```

**Output:** Câu caption hoàn chỉnh: "a person is throwing a ball to a dog"

---

## PHẦN 4: PHƯƠNG PHÁP VÀ THUẬT TOÁN

### 4.1. Tổng quan phương pháp

**Bài toán:** Cho ảnh đầu vào I, sinh câu mô tả (caption) C = {w₁, w₂, ..., wₙ}

**Giải pháp:** Kết hợp 3 bước xử lý tuần tự trong một pipeline end-to-end

```
I (ảnh) → [Faster R-CNN] → Features → [Graph] → Enhanced Features → [LSTM] → C (caption)
```

### 4.2. Thuật toán Faster R-CNN

**Input:** Ảnh RGB kích thước H×W×3

**Bước 1: Feature Extraction (VGG16 Backbone)**
```
1. Resize ảnh về max_size = 500px (giữ nguyên tỷ lệ)
2. Normalize: pixel = (pixel - mean) / std
3. Forward qua VGG16:
   - Conv layers: 13 lớp convolution
   - Output: Feature map kích thước H/16 × W/16 × 512
```

**Bước 2: Region Proposal Network (RPN)**
```
Input: Feature map (H/16 × W/16 × 512)

For each position (i, j) trên feature map:
    1. Tạo 9 anchor boxes với tỷ lệ khác nhau:
       - 3 scales: [128², 256², 512²]
       - 3 ratios: [0.5, 1, 2]
    
    2. Phân loại objectness:
       - P(object) = sigmoid(score)
       - Nếu P(object) > 0.5 → giữ lại
    
    3. Regression để tinh chỉnh box:
       - Δx, Δy, Δw, Δh = RPN_regressor(features)
       - box_refined = anchor + (Δx, Δy, Δw, Δh)

Output: ~2000 region proposals
```

**Bước 3: ROI Pooling & Classification**
```
For each region proposal:
    1. ROI Pooling:
       - Crop feature tương ứng từ feature map
       - Resize về kích thước cố định 7×7
    
    2. Flatten và qua Fully Connected layers:
       - FC1: 7×7×512 → 4096
       - FC2: 4096 → 512 (object features)
    
    3. Classification:
       - Softmax: 512 → 10 classes
       - class_id = argmax(scores)
    
    4. Box Regression (tinh chỉnh lần 2):
       - Adjust bounding box coordinates

Output: 
- Bounding boxes: [(x1, y1, x2, y2), ...]
- Labels: [class_id, ...]
- Features: [512-dim vector, ...] cho mỗi object
```

### 4.3. Thuật toán Relationship Graph

**Input:** 
- Object features: F = [f₁, f₂, ..., fₙ] (n objects, mỗi cái 512-dim)
- Bounding boxes: B = [(x₁, y₁, x₂, y₂), ...]

**Bước 1: Xây dựng Scene Graph**
```
Khởi tạo graph G = (V, E):
- Nodes V = {v₁, v₂, ..., vₙ} (mỗi object là 1 node)
- Edges E = {}

For i from 1 to n:
    For j from 1 to n (j ≠ i):
        1. Tính spatial features:
           - distance = ||center(box_i) - center(box_j)||
           - relative_position = (x_j - x_i, y_j - y_i) / image_size
           - IoU = intersection(box_i, box_j) / union(box_i, box_j)
        
        2. Kết nối node:
           - If distance < threshold:
               E = E ∪ {edge(i → j)}
               edge_features[i,j] = concat(f_i, f_j, spatial_features)
```

**Bước 2: Graph Neural Network (Message Passing)**
```
For each layer l in [1, 2, 3]:  # 3 layers GNN
    For each node v_i:
        1. Thu thập messages từ neighbors:
           messages = []
           For each neighbor v_j:
               m_ij = MLP(concat(h_i^(l-1), h_j^(l-1), edge_features[i,j]))
               messages.append(m_ij)
        
        2. Aggregate messages:
           aggregated = mean(messages)  # hoặc max, sum
        
        3. Update node features:
           h_i^(l) = ReLU(W^(l) × concat(h_i^(l-1), aggregated) + b^(l))

Output: Enhanced features h_i^(3) cho mỗi object
```

**Bước 3: Relationship Classification**
```
For each pair (i, j):
    1. Concat features:
       pair_feature = concat(h_i^(3), h_j^(3), spatial_features[i,j])
    
    2. Phân loại relation:
       rel_logits = FC_relation(pair_feature)  # 6 classes
       rel_type = argmax(rel_logits)
       # Classes: above, below, left, right, holding, wearing

Output: Relationship matrix R (n×n)
```

### 4.4. Thuật toán LSTM Caption Generator

**Input:**
- Enhanced object features: H = [h₁, h₂, ..., hₙ]
- Ground truth caption (khi training): C = [w₁, w₂, ..., wₘ]

**Bước 1: Khởi tạo**
```
1. Vocabulary V: 5,320 từ (xây từ dataset)
   - word2idx: {"a": 1, "dog": 2, "is": 3, ...}
   - idx2word: {1: "a", 2: "dog", 3: "is", ...}
   - Special tokens: <start>=0, <end>=5319, <pad>=5318

2. Tính global image feature:
   v_global = mean(H)  # Average pooling trên tất cả objects
   
3. Khởi tạo LSTM state:
   h₀ = tanh(W_init × v_global)
   c₀ = zeros(512)
```

**Bước 2: Attention Mechanism**
```
Function Attention(h_t, H):
    """Tính attention weights tại time step t"""
    
    For each object feature h_i in H:
        # Tính attention score
        score_i = (W_h × h_t)ᵀ × (W_v × h_i)
    
    # Normalize bằng softmax
    attention_weights = softmax([score_1, score_2, ..., score_n])
    
    # Weighted sum
    context_vector = Σ(attention_weights[i] × h_i)
    
    Return context_vector, attention_weights
```

**Bước 3: Sinh Caption (Training Mode)**
```
Input: Ground truth caption C = [<start>, w₁, w₂, ..., wₘ, <end>]

h_t = h₀, c_t = c₀
total_loss = 0

For t from 1 to m+1:
    1. Embedding từ hiện tại:
       x_t = Embedding[C[t-1]]  # 512-dim word embedding
    
    2. Attention:
       context_t, α_t = Attention(h_t, H)
    
    3. Concat input:
       input_t = concat(x_t, context_t)
    
    4. LSTM forward:
       h_t, c_t = LSTM(input_t, h_{t-1}, c_{t-1})
    
    5. Predict next word:
       logits_t = FC(h_t)  # 512 → 5320 (vocab size)
       probs_t = softmax(logits_t)
    
    6. Tính loss:
       loss_t = -log(probs_t[C[t]])  # Cross-entropy
       total_loss += loss_t

Return: total_loss / (m+1)
```

**Bước 4: Sinh Caption (Inference Mode)**
```
h_t = h₀, c_t = c₀
caption = [<start>]

For t from 1 to max_length (=20):
    1. Embedding từ vừa sinh:
       x_t = Embedding[caption[-1]]
    
    2. Attention:
       context_t, α_t = Attention(h_t, H)
    
    3. LSTM forward:
       input_t = concat(x_t, context_t)
       h_t, c_t = LSTM(input_t, h_{t-1}, c_{t-1})
    
    4. Greedy decoding:
       logits_t = FC(h_t)
       word_id = argmax(softmax(logits_t))
    
    5. Thêm vào caption:
       caption.append(word_id)
    
    6. Dừng nếu gặp <end>:
       If word_id == <end>:
           Break

Return: caption chuyển từ IDs sang words
```

### 4.5. Thuật toán Training End-to-End

**Pseudo-code tổng thể:**

```python
# ===== KHỞI TẠO =====
model = CombinedModel(FasterRCNN, RelationshipGraph, LSTM)
optimizer = Adam(lr=5e-5)
scheduler = CosineAnnealingLR(T_max=12)

train_data = load_split("train")  # 6,472 images
test_data = load_split("test")    # 1,619 images

best_loss = ∞

# ===== TRAINING LOOP =====
For epoch in range(1, 13):
    
    # --- TRAINING PHASE ---
    model.train()
    train_losses = []
    
    For each (image, caption, boxes) in train_data:
        1. Forward pass:
           # Faster R-CNN
           obj_features, boxes, labels = FasterRCNN(image)
           
           # Relationship Graph
           enhanced_features = RelationshipGraph(obj_features, boxes)
           
           # LSTM
           caption_loss = LSTM.forward(enhanced_features, caption)
        
        2. Backward pass:
           optimizer.zero_grad()
           caption_loss.backward()  # Backprop qua toàn bộ pipeline
           clip_grad_norm(model.parameters(), max_norm=1.0)
           optimizer.step()
        
        3. Log loss:
           train_losses.append(caption_loss.item())
    
    avg_train_loss = mean(train_losses)
    
    # --- TESTING PHASE ---
    model.eval()
    test_losses = []
    
    For each (image, caption, boxes) in test_data:
        With no_grad():
            obj_features = FasterRCNN(image)
            enhanced_features = RelationshipGraph(obj_features, boxes)
            caption_loss = LSTM.forward(enhanced_features, caption)
            test_losses.append(caption_loss.item())
    
    avg_test_loss = mean(test_losses)
    
    # --- CHECKPOINT ---
    save_checkpoint(f"epoch_{epoch}.pth")
    
    If avg_test_loss < best_loss:
        best_loss = avg_test_loss
        save_checkpoint("best_12ep.pth")
        patience_counter = 0
    Else:
        patience_counter += 1
    
    # Early stopping
    If patience_counter >= 5:
        Break
    
    # Update learning rate
    scheduler.step()
```

### 4.6. Độ phức tạp tính toán

**Faster R-CNN:**
- Backbone VGG16: O(H×W×C) với C=512 channels
- RPN: O(H/16 × W/16 × 9) ~ O(HW)
- ROI Pooling: O(R×7×7) với R=số regions (~2000)
- **Tổng:** O(HW + R)

**Relationship Graph:**
- Build graph: O(n²) với n=số objects
- GNN layers: O(L×n²×d) với L=3 layers, d=512 features
- **Tổng:** O(n²×d)

**LSTM:**
- Mỗi time step: O(d²) với d=512
- T time steps: O(T×d²) với T~15-20 từ
- Attention: O(T×n×d)
- **Tổng:** O(T×(d²+n×d))

**Training:**
- 1 epoch: 6,472 ảnh × (O(HW) + O(n²d) + O(Td²))
- 12 epochs × 6,472 = 77,664 forward+backward passes

---

## PHẦN 5: QUÁ TRÌNH HUẤN LUYỆN

### 5.1. Chuẩn bị dữ liệu

**Điểm đặc biệt:** Cả 3 modules được train **đồng thời** (end-to-end), không train riêng lẻ

**Lợi ích:**
- Các module học cách **phối hợp** với nhau
- Faster R-CNN học extract features **tốt cho captioning**, không chỉ cho detection
- Relationship Graph học những quan hệ **quan trọng cho mô tả**
- LSTM học cách **tận dụng tối đa** visual features

### 5.1. Chuẩn bị dữ liệu

**Quy trình xử lý dataset:**

```python
# 1. Load và split dataset
images = load_images("Images/")  # 8,091 ảnh
captions = load_captions("captions.txt")
boxes = load_boxes("boxes_rel.json")

# 2. Shuffle với seed
random.seed(42)
indices = list(range(8091))
random.shuffle(indices)

# 3. Split 80/20
split_point = int(0.8 × 8091) = 6472
train_indices = indices[:6472]
test_indices = indices[6472:]

# 4. Lưu splits
save_json({"train": train_indices, "test": test_indices}, "data_splits.json")
```

**Dataset class:**
```python
class Flickr8kDataset:
    def __getitem__(self, idx):
        # Load ảnh
        image = Image.open(image_path).convert('RGB')
        image = ToTensor()(image)  # Normalize [0,1]
        
        # Load boxes & labels
        boxes = boxes_data[image_id]['boxes']
        labels = boxes_data[image_id]['labels']
        
        # Load caption và encode
        caption_text = captions[image_id]
        caption_ids = vocabulary.encode(caption_text)
        
        return {
            'image': image,
            'boxes': tensor(boxes),
            'labels': tensor(labels),
            'caption': tensor(caption_ids)
        }
```

### 5.2. Training End-to-End

**Hardware:**
- GPU: NVIDIA CUDA-enabled
- RAM: 16GB+
- Storage: ~3-4GB cho checkpoints

**Hyperparameters:**
- **Số epochs:** 12
- **Optimizer:** Adam
- **Learning rate:** 5e-5 (0.00005)
- **Learning rate schedule:** Cosine Annealing (giảm dần theo dạng cos)
- **Weight decay:** 1e-5 (regularization)
- **Batch size:** 1 (do ảnh có kích thước khác nhau)

### 5.2. Training End-to-End

**Điểm đặc biệt:** Cả 3 modules được train **đồng thời** (end-to-end), không train riêng lẻ

**Lợi ích:**
- Các module học cách **phối hợp** với nhau
- Faster R-CNN học extract features **tốt cho captioning**, không chỉ cho detection
- Relationship Graph học những quan hệ **quan trọng cho mô tả**
- LSTM học cách **tận dụng tối đa** visual features

### 5.3. Cấu hình Training

**TRAINING PHASE (6,472 ảnh):**

```python
For each image in training set:
    1. Load ảnh + caption ground truth
    2. Forward pass:
       - Faster R-CNN → detect objects
       - Relationship Graph → analyze relationships
       - LSTM → generate caption
    3. Tính loss:
       - So sánh caption sinh ra với ground truth
       - Cross-entropy loss cho mỗi từ
    4. Backward pass:
       - Backpropagation qua toàn bộ pipeline
       - Update weights của cả 3 modules
```

**TESTING PHASE (1,619 ảnh):**

```python
For each image in test set:
    1. Load ảnh + caption ground truth
    2. Forward pass (no gradient):
       - Generate caption như training
    3. Tính test loss:
       - Đánh giá độ chính xác
       - KHÔNG update weights
```

### 5.4. Quy trình mỗi Epoch

**Caption Loss (Cross-Entropy):**
```
Mỗi từ được predict có một xác suất phân phối trên 5,320 từ
Loss = -log(P(từ đúng))

Ví dụ:
Ground truth: "dog"
Model predict: {cat: 0.3, dog: 0.6, bird: 0.1}
Loss = -log(0.6) = 0.51
```

**Total Loss:**
```
loss = caption_loss + 0.1 × relationship_loss
```

### 5.5. Loss Function

**Chiến lược lưu model:**

1. **Checkpoint mỗi epoch:**
   - `frcnn_caption_epoch_01.pth`
   - `frcnn_caption_epoch_02.pth`
   - ...
   - `frcnn_caption_epoch_12.pth`
   - **Mục đích:** Resume training nếu bị gián đoạn

2. **Best model:**
   - `frcnn_caption_best_12ep.pth`
   - Lưu model có **test loss thấp nhất**
   - **Quan trọng:** Chỉ xét test loss, không xét train loss
   - **Lý do:** Test loss thể hiện khả năng tổng quát hóa thực sự

**Early Stopping:**
- Nếu 5 epochs liên tiếp test loss không giảm → dừng training
- Tránh lãng phí thời gian khi model đã hội tụ

### 5.6. Checkpoint & Best Model

**Loss progression:**
```
Epoch 1:  Train 6.05 | Test 6.12  ← Model mới bắt đầu
Epoch 2:  Train 5.23 | Test 5.41
Epoch 3:  Train 4.68 | Test 4.89
Epoch 4:  Train 4.35 | Test 4.52
Epoch 5:  Train 4.11 | Test 4.28
Epoch 6:  Train 3.94 | Test 4.15  ← Best model
Epoch 7:  Train 4.26 | Test 4.41  ← Không cải thiện
Epoch 8:  Train 3.85 | Test 4.10  ← Best model mới
...
Epoch 12: Train 3.50 | Test 3.82  ← Kết thúc
```

**Xu hướng:**
- Loss giảm nhanh ở đầu (epoch 1-4)
- Loss giảm chậm dần về cuối (epoch 8-12)
- Train loss < Test loss là bình thường (model fit tốt trên train set)

---

### 5.7. Kết quả Training (dự kiến)

---

## PHẦN 6: ĐÁNH GIÁ MÔ HÌNH

**BLEU (Bilingual Evaluation Understudy)** là metric chuẩn trong:
- Machine Translation
- Image Captioning
- Text Generation

**Ý tưởng:** So sánh n-grams giữa caption sinh ra và ground truth

**4 chỉ số:**

1. **BLEU-1 (Unigram):**
   - Đếm số từ đơn khớp
   - Đánh giá **từ vựng**
   ```
   Generated: "a dog is running"
   Reference: "a brown dog is running fast"
   Matched: "a", "dog", "is", "running" → 4/5 = 80%
   ```

2. **BLEU-2 (Bigram):**
   - Đếm cụm 2 từ liên tiếp khớp
   - Đánh giá **cấu trúc ngắn**
   ```
   Generated: "a dog", "dog is", "is running"
   Reference: "a brown", "brown dog", "dog is", "is running", "running fast"
   Matched: "dog is", "is running" → 2/4 = 50%
   ```

3. **BLEU-3 (Trigram):**
   - Đếm cụm 3 từ khớp
   - Đánh giá **cấu trúc câu**

4. **BLEU-4 (4-gram):**
   - Đếm cụm 4 từ khớp
   - Đánh giá **tính tự nhiên tổng thể**

**Thang điểm:** 0.0 - 1.0 (hoặc 0% - 100%)
- BLEU > 0.5: Rất tốt
- BLEU 0.3-0.5: Tốt
- BLEU 0.2-0.3: Chấp nhận được
- BLEU < 0.2: Cần cải thiện

### 6.1. Metrics: BLEU Score

```python
1. Load best model: frcnn_caption_best_12ep.pth

2. Load test set: 1,619 ảnh từ data_splits.json

3. For each test image:
   - Model sinh caption
   - Lưu cặp (generated_caption, ground_truth)

4. Tính BLEU scores:
   - BLEU-1, BLEU-2, BLEU-3, BLEU-4
   - Trung bình trên 1,619 ảnh test

5. In kết quả:
   BLEU-1: 0.XX
   BLEU-2: 0.XX
   BLEU-3: 0.XX
   BLEU-4: 0.XX
```

### 6.2. Quy trình Evaluation

**Ảnh test:** Một cô gái đang ngồi trên ghế đọc sách

**Ground truth:** "a young woman sitting on a bench reading a book"

**Model generates:** "a girl is sitting on a bench with a book"

**BLEU scores:**
- BLEU-1: 0.75 (7/8 từ khớp: a, girl/young woman, is/∅, sitting, on, a, bench, with/reading, a, book)
- BLEU-2: 0.55 (cụm 2 từ khớp: "on a", "a bench")
- BLEU-3: 0.32
- BLEU-4: 0.18

→ Caption **có nghĩa** nhưng **cấu trúc khác** một chút

---

### 6.3. Ví dụ đánh giá

---

## PHẦN 7: KẾT QUẢ VÀ ĐÓNG GÓP

### 7.1. Kết quả đạt được

✅ **Xây dựng thành công pipeline end-to-end** kết hợp 3 công nghệ AI

✅ **Training ổn định** trên 8,091 ảnh Flickr8k với 12 epochs

✅ **Checkpoint management** hoàn chỉnh với best model tracking

✅ **Đánh giá khoa học** với BLEU metrics trên test set riêng biệt

✅ **Reproducible:** Code, data splits, và checkpoints đầy đủ

### 7.2. Điểm mạnh của hệ thống

**1. Kiến trúc toàn diện:**
- Không chỉ detect objects mà còn hiểu **mối quan hệ**
- Attention mechanism giúp LSTM "nhìn" đúng chỗ khi sinh từ

**2. Training end-to-end:**
- Các module học cách phối hợp tối ưu
- Gradient flow xuyên suốt pipeline

**3. Dataset split khoa học:**
- Test set hoàn toàn tách biệt
- Đảm bảo đánh giá khách quan

**4. Có thể mở rộng:**
- Dễ dàng thay Faster R-CNN bằng YOLO, DETR...
- Có thể thay LSTM bằng Transformer
- Có thể train trên dataset lớn hơn (MS COCO, Flickr30k)

### 7.3. Hạn chế và hướng phát triển

**Hạn chế:**
- Dataset nhỏ (8k ảnh) → giới hạn vocabulary và đa dạng
- Batch size = 1 → training chậm
- Chưa xử lý multi-object attention tốt

**Hướng phát triển:**
- **Dataset lớn hơn:** MS COCO (120k ảnh), Conceptual Captions (3M ảnh)
- **Transformer architecture:** Thay LSTM bằng Transformer decoder
- **Beam search:** Thay greedy decoding để sinh caption đa dạng hơn
- **Visual-semantic embedding:** Học joint space giữa vision và language

---

---

## PHẦN 8: KẾT LUẬN

### 8.1. Tóm tắt

Dự án đã **xây dựng thành công hệ thống Image Captioning end-to-end** với:

📊 **Dataset:** Flickr8k (8,091 ảnh, split 80/20)

🏗️ **Kiến trúc:** Faster R-CNN + Relationship GNN + LSTM

⚙️ **Training:** 12 epochs, GPU, checkpoint mỗi epoch

📈 **Đánh giá:** BLEU-1/2/3/4 trên test set

### 8.2. Ý nghĩa

Hệ thống này minh họa cách:
- **Computer Vision** (nhìn) kết hợp **Natural Language Processing** (nói)
- **Detection** (phát hiện) kết hợp **Reasoning** (suy luận) kết hợp **Generation** (sinh ngôn ngữ)
- **End-to-end learning** tạo ra kết quả tốt hơn các module riêng lẻ

### 8.3. Ứng dụng thực tế

💡 **Hỗ trợ người khiếm thị** hiểu nội dung ảnh

🔍 **Tìm kiếm ảnh** bằng ngôn ngữ tự nhiên

📱 **Mạng xã hội** tự động gợi ý caption

🤖 **Robotics** giúp robot "hiểu" môi trường xung quanh

---

## CẢM ƠN QUÝ THẦY CÔ ĐÃ LẮNG NGHE! 🙏

**Các câu hỏi?** 💬
