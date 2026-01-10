# PHASE 3 QUICK START GUIDE

## What is Phase 3?

Phase 3 is the **psychological reasoning engine** that combines:
- 👁️ **Face emotion** (Phase 1)
- 🎤 **Voice emotion + stress** (Phase 2)
- 🧠 **Psychological fusion** (Phase 3)

Into **meaningful psychological insights**.

---

## Installation

### Requirements
```bash
# Already installed if you completed Phase 1 & 2
pip install torch torchvision opencv-python numpy sounddevice librosa
```

### Verify Installation
```bash
python -c "from inference.phase3_multimodal_fusion import Phase3MultiModalFusion; print('✅ Phase 3 installed')"
```

---

## Quick Test (5 minutes)

### Test 1: Run Scenarios
```bash
python inference/phase3_multimodal_fusion.py
```

**What you'll see:**
- 4 psychological scenarios tested
- Detailed analysis for each
- Mental state inferences
- Risk assessments

**Expected output:**
```
SCENARIO 1: Person smiling but stressed
→ Mental State: HAPPY UNDER STRESS
→ Risk Level: HIGH
→ Reasoning: "Happiness detected but stress is high → masked emotion"
```

### Test 2: Run Integrated System
```bash
python inference/integrated_psychologist_ai.py
```

**What happens:**
- Opens webcam (Phase 1: face detection)
- Captures microphone (Phase 2: voice + stress)
- Shows real-time psychological state (Phase 3: fusion)

**GUI shows:**
- Live video with face detection
- Info panel with psychological state
- Risk level and stability
- Explanations for decisions

**Controls:**
- `q` - Quit
- `s` - Save screenshot
- `p` - Pause/Resume
- `i` - Toggle info panel

---

## Usage Examples

### Example 1: Basic Usage

```python
from inference.phase3_multimodal_fusion import Phase3MultiModalFusion

# Initialize Phase 3
phase3 = Phase3MultiModalFusion()

# Process a frame
state = phase3.process_frame(
    face_emotion='happy',
    face_confidence=0.75,
    face_detected=True,
    voice_emotion='happy',
    voice_confidence=0.65,
    audio_quality=0.9,
    stress_level='high',
    stress_confidence=0.82
)

# Print results
print(f"Mental State: {state.mental_state.value}")
print(f"Dominant Emotion: {state.dominant_emotion}")
print(f"Risk: {state.risk_level.value}")
print(f"Confidence: {state.confidence*100:.1f}%")
print(f"Stability: {state.stability_score*100:.1f}%")

# Print reasoning
for explanation in state.explanations:
    print(f"  • {explanation}")
```

**Output:**
```
Mental State: happy_under_stress
Dominant Emotion: happy
Risk: high
Confidence: 24.9%
Stability: 100.0%
  • High stress detected (82% confidence)
  • Happiness detected but stress is high → masked emotion
  • Mental state: happy_under_stress
  • Pattern: High stress persisting (100% of time)
```

### Example 2: Integrate with Existing Code

```python
from inference.phase3_multimodal_fusion import Phase3MultiModalFusion

# Your existing Phase 1 & 2 code
face_detector = YourFaceDetector()
voice_detector = YourVoiceDetector()

# Add Phase 3
phase3 = Phase3MultiModalFusion()

# In your processing loop
for frame in video:
    # Phase 1
    face_emotion, face_conf = face_detector.predict(frame)
    
    # Phase 2
    voice_emotion, voice_conf = voice_detector.predict(audio)
    stress_level, stress_conf = voice_detector.predict_stress(audio)
    
    # Phase 3
    state = phase3.process_frame(
        face_emotion=face_emotion,
        face_confidence=face_conf,
        face_detected=True,
        voice_emotion=voice_emotion,
        voice_confidence=voice_conf,
        audio_quality=0.9,
        stress_level=stress_level,
        stress_confidence=stress_conf
    )
    
    # Use results
    if state.risk_level.value in ['high', 'critical']:
        send_alert(state)
```

---

## Understanding Output

### PsychologicalState Object

```python
state = phase3.process_frame(...)

# Access fields
state.dominant_emotion      # Primary emotion: 'happy', 'sad', etc.
state.hidden_emotion        # Masked emotion or None
state.mental_state          # MentalState enum (e.g., HAPPY_UNDER_STRESS)
state.confidence            # 0.0 to 1.0
state.risk_level            # RiskLevel enum (LOW, MODERATE, HIGH, CRITICAL)
state.stability_score       # 0.0 (unstable) to 1.0 (stable)
state.explanations          # List of reasoning strings
state.temporal_patterns     # Detected patterns (stress persistence, etc.)
```

### Mental States

| State | Meaning |
|-------|---------|
| `CALM` | Relaxed, stable, no stress |
| `JOYFUL` | Genuinely happy |
| `HAPPY_UNDER_STRESS` | Smiling but stressed |
| `STRESSED` | Under pressure |
| `ANXIOUS` | Fear + high stress |
| `OVERWHELMED` | Multiple stressors |
| `EMOTIONALLY_MASKED` | Hiding emotions |
| `EMOTIONALLY_UNSTABLE` | Rapid mood swings |
| `SAD_DEPRESSED` | Deep sadness |
| `ANGRY_STRESSED` | Anger + pressure |
| `FEARFUL` | Afraid |

### Risk Levels

| Level | Meaning | Action |
|-------|---------|--------|
| `LOW` | Normal state | No action needed |
| `MODERATE` | Some concern | Monitor |
| `HIGH` | Significant concern | Check in with person |
| `CRITICAL` | Urgent | Immediate intervention |

---

## Common Use Cases

### Use Case 1: Mental Health Monitoring
```python
# Track psychological state over time
states = []
for frame in session:
    state = phase3.process_frame(...)
    states.append(state)

# Analyze session
avg_stability = np.mean([s.stability_score for s in states])
max_risk = max([s.risk_level for s in states])
```

### Use Case 2: Alert System
```python
def check_alerts(state):
    if state.risk_level == RiskLevel.CRITICAL:
        send_immediate_alert()
    elif state.risk_level == RiskLevel.HIGH:
        schedule_check_in()
    elif state.mental_state == MentalState.EMOTIONALLY_MASKED:
        log_masking_event()
```

### Use Case 3: Self-Awareness Tool
```python
# Show user their psychological state
print(format_psychological_state(state))

# Give feedback
if state.mental_state == MentalState.STRESSED:
    suggest_breathing_exercise()
elif state.mental_state == MentalState.HAPPY_UNDER_STRESS:
    suggest_break()
```

---

## Troubleshooting

### Issue: Low confidence (<30%)

**Causes:**
- Face not detected clearly
- Audio quality poor
- Conflicting signals (face vs voice)

**Solutions:**
- Improve lighting for face detection
- Use better microphone
- Check if person is actually masking emotions (low confidence expected)

### Issue: Mental state doesn't make sense

**Debug steps:**
1. Check explanations: `state.explanations`
2. Check raw signals: `state.raw_signals`
3. Check temporal patterns: `state.temporal_patterns`
4. Verify input data quality

### Issue: Risk level seems wrong

**Remember:**
- Risk is **psychological**, not accuracy-based
- `HAPPY_UNDER_STRESS` = HIGH risk (person hiding stress)
- `SAD_DEPRESSED` = HIGH risk (mental health concern)
- Risk doesn't mean prediction is wrong

---

## Performance Tips

### Optimize Speed

```python
# Process every N frames (not every frame)
if frame_count % 3 == 0:  # Process every 3rd frame
    state = phase3.process_frame(...)
else:
    use_previous_state()
```

### Reduce Memory

```python
# Use smaller temporal window
phase3 = Phase3MultiModalFusion()
phase3.memory.window_size = 15  # Default is 30
```

### GPU Acceleration

Phase 3 is lightweight (rule-based), but Phase 1 & 2 use GPU:
```python
# Ensure models are on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## Next Steps

### Learn More
- Read full documentation: `docs/PHASE3_DOCUMENTATION.md`
- Study code: `inference/phase3_multimodal_fusion.py`
- Understand mental states: See Layer 4 in documentation

### Customize
- Add new mental states (edit `MentalState` enum)
- Tune fusion rules (edit `FusionEngine`)
- Adjust risk thresholds (edit `RiskAssessor`)

### Integrate
- Add to existing application
- Build alert system
- Create therapy tool
- Make self-awareness app

---

## FAQ

**Q: Do I need Phase 1 and Phase 2 to use Phase 3?**
A: Phase 3 needs their **outputs**, but you can use any face/voice detector. Just format inputs correctly.

**Q: Is Phase 3 accurate?**
A: Phase 3 uses **psychological rules**, not trained models. It's interpretable but not statistically validated. Use for insights, not clinical diagnosis.

**Q: Can I use Phase 3 alone?**
A: Yes! Just provide fake or preprocessed inputs for testing.

**Q: How do I know if Phase 3 is working?**
A: Run test scenarios (`python inference/phase3_multimodal_fusion.py`). If outputs make psychological sense, it's working.

**Q: Can I train Phase 3?**
A: Currently rule-based (interpretable). Future version may add learned fusion weights.

---

## Support

**Problems?**
1. Check error messages
2. Verify inputs are correctly formatted
3. Run test scenarios
4. Read full documentation

**Questions?**
- Technical: See code comments
- Psychological: See Layer 4 documentation
- Integration: Check `integrated_psychologist_ai.py`

---

## Summary

**Phase 3 in 3 steps:**

1. **Initialize:**
   ```python
   phase3 = Phase3MultiModalFusion()
   ```

2. **Process:**
   ```python
   state = phase3.process_frame(face, voice, stress)
   ```

3. **Use:**
   ```python
   print(state.mental_state, state.risk_level)
   ```

**That's it! You now have a psychological reasoning engine.** 🧠✨

---

Run the integrated system to see it in action:
```bash
python inference/integrated_psychologist_ai.py
```
