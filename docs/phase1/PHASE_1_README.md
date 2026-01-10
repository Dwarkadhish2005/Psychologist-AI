# 🎭 PHASE 1: FACIAL EMOTION RECOGNITION

**Goal:** Train a CNN to recognize emotions from face images in real-time.

---

## 🧩 STEP 1: Define the Problem (Like a Pro)

### Task Type

👉 **Multi-class image classification**

### Input

- **Face image** (grayscale or RGB)
- **Size:** 48x48 or 64x64 pixels

### Output (Emotion Classes)

Start simple (don't be greedy 😤):

| Class | Emotion |
|-------|---------|
| 0 | Angry 😠 |
| 1 | Happy 😊 |
| 2 | Sad 😢 |
| 3 | Surprise 😲 |
| 4 | Neutral 😐 |

**5 classes = manageable + accurate.**

---

## 🗂️ STEP 2: Dataset (No Dataset = No AI)

### 🔥 Best Beginner Dataset

**FER-2013 style emotion dataset**
- Thousands of labeled face images
- Perfect for CNNs
- Grayscale = faster training

### 📁 Dataset Folder Structure (MANDATORY)

```
data/
└── face_emotion/
    ├── train/
    │   ├── angry/     (500-1000 images)
    │   ├── happy/     (500-1000 images)
    │   ├── sad/       (500-1000 images)
    │   ├── surprise/  (500-1000 images)
    │   └── neutral/   (500-1000 images)
    ├── val/
    │   ├── angry/
    │   ├── happy/
    │   ├── sad/
    │   ├── surprise/
    │   └── neutral/
    └── test/
        ├── angry/
        ├── happy/
        ├── sad/
        ├── surprise/
        └── neutral/
```

### ⚠️ Rule

- **Train** → learning
- **Val** → ego check
- **Test** → final exam

### 📥 Where to Get Data

**Option 1: Kaggle FER-2013**
```bash
# Download from: https://www.kaggle.com/datasets/msambare/fer2013
# Or use Kaggle API:
kaggle datasets download -d msambare/fer2013
```

**Option 2: CK+ Dataset**
- [Extended Cohn-Kanade Dataset](http://www.consortium.ri.cmu.edu/ckagree/)

**Option 3: JAFFE Dataset**
- Japanese Female Facial Expression (JAFFE)

---

## 🧼 STEP 3: Preprocessing (Glow-up for Images ✨)

Every image must:
1. ✅ Be **face-only** (no background)
2. ✅ **Resized** to same shape
3. ✅ **Normalized** (0–1)

### Preprocessing Logic

```
Image
  → Grayscale
  → Resize (48x48)
  → Normalize (pixel / 255.0)
  → Tensor
```

### 🔑 Critical Rule

This **SAME preprocessing** must be used during:
- ✅ Training
- ✅ Validation
- ✅ Webcam inference

**Consistency = accuracy** 📈

### Example Code

```python
import cv2
import numpy as np

def preprocess_face(image, target_size=(48, 48)):
    """
    Standard preprocessing for facial emotion recognition.
    
    Args:
        image: Input image (BGR or RGB)
        target_size: Target dimensions (width, height)
    
    Returns:
        Preprocessed image tensor
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize
    resized = cv2.resize(gray, target_size)
    
    # Normalize to [0, 1]
    normalized = resized.astype('float32') / 255.0
    
    # Add channel dimension (1, 48, 48) for PyTorch
    tensor = np.expand_dims(normalized, axis=0)
    
    return tensor
```

---

## 🧠 STEP 4: CNN Architecture (Simple but Deadly)

We start with a **basic CNN** — no overengineering.

### 🧱 Architecture Intuition

```
Input (48x48x1)
    ↓
Conv → ReLU → MaxPool
    ↓
Conv → ReLU → MaxPool
    ↓
Flatten
    ↓
Dense → ReLU → Dropout
    ↓
Dense → Softmax (5 classes)
```

### Why This Works

- **CNN learns facial features**
- **Early layers** → edges, eyes, mouth
- **Deep layers** → expressions

### PyTorch Implementation

```python
import torch.nn as nn

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(EmotionCNN, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 48x48 → 24x24
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 24x24 → 12x12
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # 12x12 → 6x6
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x
```

### 🎯 Model Summary

- **Parameters:** ~300K (lightweight)
- **Input:** (batch, 1, 48, 48)
- **Output:** (batch, 5) logits

💡 **Deep models later. First master THIS.**

---

## 🏋️ STEP 5: Training the Model (Actual AI Birth 👶🧠)

### Key Training Choices

| Component | Choice | Why |
|-----------|--------|-----|
| **Loss Function** | Categorical Cross-Entropy | Multi-class classification |
| **Optimizer** | Adam | Fast + stable + W optimizer |
| **Learning Rate** | 0.001 | Standard starting point |
| **Epochs** | 30–50 | Stop early if val loss plateaus |
| **Batch Size** | 32 or 64 | Balance speed & memory |

### 🔁 Training Loop (Mental Model)

Every epoch:

```
1. Forward pass (predictions)
2. Loss calculation (how wrong?)
3. Backpropagation (compute gradients)
4. Weight update (learn from mistakes)
5. Validation check (prevent overfitting)
6. Repeat until model stops improving
```

**This loop = heart of deep learning** ❤️

### Training Script Flow

```python
# 1. Load data
train_loader, val_loader = load_emotion_data()

# 2. Build model
model = EmotionCNN(num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    # Save best model
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), 'models/emotion_cnn_best.pth')
        best_val_acc = val_acc

# 4. Test
test_acc = evaluate(model, test_loader)
```

---

## 📊 STEP 6: Evaluation (Reality Check 😬)

You **MUST** evaluate using:

### 1. Accuracy

```python
correct = (predictions == labels).sum()
accuracy = 100 * correct / total
```

### 2. Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_labels, predicted_labels)
```

### Why Both?

- **Accuracy lies sometimes**
- **Confusion matrix exposes bias**

### Example Issue

```
Model confuses sad with neutral
→ Dataset imbalance issue
→ Need more sad images or data augmentation
```

### Target Metrics

| Metric | Target |
|--------|--------|
| **Training Accuracy** | ~80-90% |
| **Validation Accuracy** | **65-75%** (acceptable for Phase 1) |
| **Test Accuracy** | ~65-75% |

⚠️ If train_acc >> val_acc → **overfitting**

---

## 💾 STEP 7: Save Model & Labels (DO NOT SKIP)

You will save:

1. ✅ **Model weights** (`.pth`)
2. ✅ **Class index mapping** (`labels.json`)
3. ✅ **Config** (`config.json`)

### Example Structure

```
models/
└── face_emotion/
    ├── emotion_cnn.pth          # Model weights
    ├── labels.json              # {0: 'angry', 1: 'happy', ...}
    └── config.json              # Model hyperparameters
```

### Why?

Because later:
- **Fusion module** needs labels
- **Inference** must match training
- **Future-you** will thank present-you 🫂

### Save Code

```python
import torch
import json

# Save model
torch.save(model.state_dict(), 'models/face_emotion/emotion_cnn.pth')

# Save labels
labels = {0: 'angry', 1: 'happy', 2: 'sad', 3: 'surprise', 4: 'neutral'}
with open('models/face_emotion/labels.json', 'w') as f:
    json.dump(labels, f, indent=4)

# Save config
config = {
    'input_size': [1, 48, 48],
    'num_classes': 5,
    'architecture': 'EmotionCNN'
}
with open('models/face_emotion/config.json', 'w') as f:
    json.dump(config, f, indent=4)
```

---

## 🎥 STEP 8: Real-Time Webcam Inference (GOOSEBUMPS 🔥)

### Pipeline

```
Webcam
  → Face Detection (Haar Cascade / MediaPipe)
  → Crop Face ROI
  → Preprocess (same as training!)
  → CNN Prediction
  → Display Emotion Label on Screen
```

### 🔑 Critical Distinction

**Face detection ≠ emotion detection**
- **Separate tasks, separate logic**
- ⚠️ Only feed **face ROI** to CNN

### Implementation Flow

```python
import cv2

# Load model
model = EmotionCNN(num_classes=5)
model.load_state_dict(torch.load('models/face_emotion/emotion_cnn.pth'))
model.eval()

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Webcam loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Crop face
        face_roi = gray[y:y+h, x:x+w]
        
        # Preprocess
        preprocessed = preprocess_face(face_roi)
        
        # Predict
        with torch.no_grad():
            output = model(torch.tensor(preprocessed))
            emotion = labels[output.argmax().item()]
        
        # Display
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🧪 STEP 9: Improve Accuracy (Like a Scientist 🧪)

### If accuracy < 65%:

| Problem | Solution |
|---------|----------|
| Low accuracy | Add data augmentation (flip, rotate, brightness) |
| Class imbalance | Balance dataset or use weighted loss |
| Overfitting | Add dropout, reduce model complexity |
| Underfitting | Increase image size (64x64), add layers |
| Slow convergence | Adjust learning rate, use scheduler |

### Data Augmentation Example

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

### 🎯 Rule

**Bad accuracy = data problem 80% of the time**

---

## ✅ PHASE 1 CHECKPOINT (YOU MUST HIT THIS)

Before Phase 2, you should have:

- ✔ **CNN trained by YOU** (not pretrained)
- ✔ **Validation accuracy ~65–75%**
- ✔ **Webcam emotion detection working**
- ✔ **Model + labels saved**
- ✔ **Training plots saved in `reports/`**

### 🎮 Boss-Level Challenge

**Test your model on:**
1. Your own face (different emotions)
2. Friends/family
3. YouTube videos (extract frames)

If it works → **portfolio-worthy** 🏆

---

## 📚 Resources

### Datasets
- [FER-2013 (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013)
- [AffectNet](http://mohammadmahoor.com/affectnet/)
- [CK+ Dataset](http://www.consortium.ri.cmu.edu/ckagree/)

### Tutorials
- [PyTorch Image Classification](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [OpenCV Face Detection](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)

### Papers
- [FER+ Paper](https://arxiv.org/abs/1608.01041)
- [Emotion Recognition Survey](https://arxiv.org/abs/2005.00822)

---

## 🚀 Next Phase Preview

**Phase 2: Voice Emotion Analysis**
- Audio feature extraction (MFCCs, spectrograms)
- RNN/LSTM for temporal patterns
- Multi-modal fusion (face + voice)

**Stay locked in.** 🔒
