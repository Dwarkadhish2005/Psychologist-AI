# Models Directory

This directory stores trained model weights and configurations.

## Structure

```
models/
├── README.md                    # This file
├── emotion_face_cnn.pth        # Facial emotion model weights
├── emotion_face_config.json    # Model configuration
├── emotion_face_labels.json    # Class mappings {0: 'happy', 1: 'sad', ...}
├── emotion_voice_model.pth     # Voice emotion model weights
└── ...
```

## Important Notes

⚠️ **Model files (`.pth`, `.h5`, `.pkl`) are NOT committed to Git** (they're too large).

✅ **Config files (`.json`) ARE committed** so you know how to load the models.

## Saving Models

When you train a model, ALWAYS save:

1. **Model weights** - `model.pth`
2. **Config** - `model_config.json` (input size, architecture details)
3. **Labels** - `model_labels.json` (class mappings)

### Example Save Code

```python
import torch
import json

# Save model weights
torch.save(model.state_dict(), 'models/emotion_cnn.pth')

# Save config
config = {
    'input_size': (48, 48),
    'num_classes': 7,
    'architecture': 'ResNet18'
}
with open('models/emotion_cnn_config.json', 'w') as f:
    json.dump(config, f, indent=4)

# Save labels
labels = {0: 'angry', 1: 'happy', 2: 'sad', 3: 'neutral', 4: 'fear', 5: 'surprise', 6: 'disgust'}
with open('models/emotion_cnn_labels.json', 'w') as f:
    json.dump(labels, f, indent=4)
```

## Loading Models

```python
import torch
import json

# Load config
with open('models/emotion_cnn_config.json', 'r') as f:
    config = json.load(f)

# Build model
model = EmotionCNN(num_classes=config['num_classes'])

# Load weights
model.load_state_dict(torch.load('models/emotion_cnn.pth'))
model.eval()

# Load labels
with open('models/emotion_cnn_labels.json', 'r') as f:
    labels = json.load(f)
```

---

**Keep your models organized. Future-you will thank you.** 🙏
