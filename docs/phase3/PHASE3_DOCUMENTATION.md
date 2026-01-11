# PHASE 3: MULTI-MODAL FUSION & PSYCHOLOGICAL REASONING

## Overview

Phase 3 is the **brain** of the Psychologist AI system. It combines face emotion, voice emotion, and stress detection into psychological insights that go beyond simple emotion classification.

**What makes Phase 3 special:**
- It doesn't just detect emotions—it **understands psychological states**
- It detects **hidden emotions** (what people try to hide)
- It provides **explainable reasoning** for every decision
- It assesses **risk levels** for mental health concerns
- It tracks **temporal patterns** (stability, masking, stress persistence)

---

## Architecture

Phase 3 consists of 4 layers:

```
┌─────────────────────────────────────────┐
│  RAW INPUTS                             │
│  • Face emotion + confidence            │
│  • Voice emotion + confidence           │
│  • Stress level + confidence            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  LAYER 1: Signal Normalization          │
│  • Reliability weighting                │
│  • Quality assessment                   │
│  • Common emotion space                 │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  LAYER 2: Temporal Reasoning            │
│  • Rolling window (30 frames)           │
│  • Pattern detection                    │
│  • Stability analysis                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  LAYER 3: Fusion Logic                  │
│  • Rule-based psychology                │
│  • Conflict resolution                  │
│  • Hidden emotion detection             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  LAYER 4: Psychological Reasoning       │
│  • Mental state inference               │
│  • Risk assessment                      │
│  • Explanation generation               │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  OUTPUT: PsychologicalState             │
│  • Dominant emotion                     │
│  • Hidden emotion                       │
│  • Mental state                         │
│  • Confidence + explanations            │
│  • Risk level + stability               │
└─────────────────────────────────────────┘
```

---

## Layer 1: Signal Normalization

### Purpose
Raw predictions from face and voice models have different:
- **Reliabilities** (voice is harder to fake than face)
- **Confidence scales** (need normalization)
- **Emotion vocabularies** (face has 7 classes, voice has 5)

### How it works

Each signal gets a **reliability score**:

| Modality | Reliability | Reasoning |
|----------|------------|-----------|
| Face | 0.5 (Medium) | People can control facial expressions (masking) |
| Voice | 0.7 (High) | Voice patterns are harder to fake consciously |
| Stress | 0.9 (Very High) | Physiological response, nearly impossible to fake |

**Weighted Confidence:**
```python
weighted_confidence = confidence × reliability × signal_quality
```
**Emotion Mapping:**
- `disgust` (face) → `angry` (voice equivalent)
- `surprise` (face) → `neutral` (transient, treated as neutral)

### Output
`NormalizedSignal` objects containing:
- Modality type
- Emotion label
- Raw confidence
- Weighted confidence
- Signal quality
- Timestamp

---

## Layer 2: Temporal Reasoning

### Purpose
Humans aren't frame-by-frame creatures. Emotions have **persistence**, **patterns**, and **transitions**.

### Memory Window
- Maintains **30 frames** of history (~15 seconds at 2 FPS)
- Tracks face, voice, and stress separately

### Pattern Detection

**1. Stress Persistence**
```
Trigger: Stress "high" for >70% of last 10 frames
Meaning: Chronic stress, not momentary spike
Risk: HIGH → CRITICAL
```

**2. Emotional Masking**
```
Trigger: Face neutral >60% + Voice emotional >60%
Meaning: Person hiding emotion with neutral face
Example: Face neutral, voice shows fear
State: EMOTIONALLY_MASKED
```

**3. Emotional Instability**
```
Trigger: >3 emotion switches in 10 frames
Meaning: Rapid mood changes, unstable state
Example: happy → sad → angry → neutral → fear
State: EMOTIONALLY_UNSTABLE
Stability: <30%
```

### Stability Score
```python
stability = 1.0 - (switch_penalty + instability_penalty)
# 0.0 = completely unstable
# 1.0 = perfectly stable
```

---

## Layer 3: Fusion Logic (Core Brain)

### Rule-Based Psychology

This is where the **magic** happens. Phase 3 uses psychology-inspired rules to combine signals intelligently.

**Rule 1: Stress Dominance**
```python
if stress == 'high' and stress_confidence > 0.5:
    # Stress overrides other emotions
    # If happy detected → masked emotion
```

**Rule 2: Voice > Face**
```python
if voice_confidence > face_confidence + 0.2:
    # Trust voice more (harder to fake)
    dominant = voice_emotion
    hidden = face_emotion if different
```

**Rule 3: Agreement Bonus**
```python
if face_emotion == voice_emotion:
    # High confidence when modalities agree
    # No conflict penalty
```

**Rule 4: Conflict Resolution**
```python
if voice != face:
    # Weighted decision
    # Hidden emotion = minority signal
    # Lower confidence (conflict penalty)
```

### Example Fusion Cases

**Case 1: Happy under stress**
```
Input:
  Face: happy (0.75)
  Voice: happy (0.65)
  Stress: high (0.82)

Output:
  Dominant: happy
  Hidden: stress
  State: HAPPY_UNDER_STRESS
  Explanation: "Happiness detected but stress is high → masked emotion"
```

**Case 2: Emotional masking**
```
Input:
  Face: neutral (0.70)
  Voice: fear (0.68)
  Stress: medium (0.60)

Output:
  Dominant: fear
  Hidden: neutral
  State: EMOTIONALLY_MASKED
  Explanation: "Face neutral, voice shows fear → masking detected"
```

---

## Layer 4: Psychological Reasoning

### Mental State Taxonomy

Phase 3 doesn't just output emotions—it infers **mental states**:

| Mental State | Meaning | Triggers |
|--------------|---------|----------|
| `CALM` | Relaxed, stable, low stress | neutral + low stress + stable |
| `JOYFUL` | Genuinely happy | happy + low stress + high confidence |
| `STRESSED` | Under pressure | any emotion + high stress |
| `HAPPY_UNDER_STRESS` | Smiling but tense | happy + high stress |
| `EMOTIONALLY_MASKED` | Hiding true feelings | face neutral + voice emotional |
| `ANXIOUS` | Fear + stress | fear + high stress |
| `OVERWHELMED` | Multiple stressors | persistent high stress + instability |
| `EMOTIONALLY_FLAT` | Low engagement | neutral + low confidence + low stress |
| `EMOTIONALLY_UNSTABLE` | Rapid mood swings | >3 switches in 10 frames |
| `ANGRY_STRESSED` | Anger + pressure | angry + high stress |
| `SAD_DEPRESSED` | Deep sadness | sad + low stress + stable |
| `FEARFUL` | Afraid without stress | fear + low/medium stress |
| `STABLE_POSITIVE` | Contentment | happy + low stress |
| `STABLE_NEGATIVE` | Persistent negativity | sad/angry/fear + stable |
| `CONFUSED` | Uncertain state | neutral + medium stress |

### Mental State Inference Logic

```python
# Priority order:
1. Check patterns (masking, instability) → MASKED or UNSTABLE
2. Check stress level:
   - high + happy → HAPPY_UNDER_STRESS
   - high + fear → ANXIOUS
   - high + angry → ANGRY_STRESSED
   - high + persistent → OVERWHELMED
3. Check positive states:
   - happy + low stress + high conf → JOYFUL
   - happy + low stress → STABLE_POSITIVE
4. Check negative states:
   - sad + low stress → SAD_DEPRESSED
   - fear + low stress → FEARFUL
5. Default: CALM or CONFUSED
```

---

## Confidence & Explanation Engine

### Confidence Calculation

```python
base_confidence = (
    face.weighted_confidence × 0.25 +
    voice.weighted_confidence × 0.35 +
    stress.weighted_confidence × 0.40
)

final_confidence = base_confidence - penalties

penalties:
  • Conflict penalty: -0.15 (if hidden emotion exists)
  • Instability penalty: -0.10 × intensity
  • Low quality penalty: -0.15 × (1 - avg_quality)
```

**Why this matters:**
- Users can **trust** high-confidence predictions
- Low confidence → system is uncertain, don't over-rely

### Explanation Generation

Every decision includes **human-readable reasoning**:

```
Explanations:
  1. High stress detected (82% confidence)
  2. Happiness detected but stress is high → masked emotion
  3. Mental state: happy_under_stress
  4. Pattern: High stress persisting (100% of time)
  5. Stability: 100%
  6. Risk: high
```

**Why this matters:**
- **Transparency**: Users understand the AI's reasoning
- **Trust**: Can verify if reasoning makes sense
- **Debugging**: Developers can trace decisions

---

## Risk & Safety Assessment

### Risk Levels

| Risk Level | Meaning | Triggers |
|------------|---------|----------|
| **LOW** | Normal, stable mental state | CALM, JOYFUL, STABLE_POSITIVE |
| **MODERATE** | Some concern, watch closely | STRESSED, MASKED, UNSTABLE, FEARFUL |
| **HIGH** | Significant concern | ANGRY_STRESSED, SAD_DEPRESSED, persistent stress |
| **CRITICAL** | Urgent attention needed | OVERWHELMED, ANXIOUS + persistent stress |

### Risk Logic

```python
if state == OVERWHELMED or (state == ANXIOUS and stress_persistent):
    return CRITICAL

if state in [ANGRY_STRESSED, SAD_DEPRESSED]:
    return HIGH

if stress == 'high' and stress_persistent:
    return HIGH

if state in [STRESSED, MASKED, UNSTABLE, FEARFUL, STABLE_NEGATIVE]:
    return MODERATE

return LOW
```

**Use cases:**
- Therapist alert system
- Self-awareness tool
- Early intervention trigger

---

## Usage Examples

### Basic Usage

```python
from inference.phase3_multimodal_fusion import Phase3MultiModalFusion

# Initialize
phase3 = Phase3MultiModalFusion()

# Process frame
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

# Access results
print(f"Mental State: {state.mental_state.value}")
print(f"Dominant Emotion: {state.dominant_emotion}")
print(f"Risk Level: {state.risk_level.value}")
print(f"Confidence: {state.confidence*100:.1f}%")
print(f"Stability: {state.stability_score*100:.1f}%")

for explanation in state.explanations:
    print(f"  - {explanation}")
```

### Integrated System

```python
# Run complete system (all 3 phases)
python inference/integrated_psychologist_ai.py
```

This runs:
- **Phase 1**: Face emotion detection (webcam)
- **Phase 2**: Voice emotion + stress detection (microphone)
- **Phase 3**: Multi-modal fusion (psychological reasoning)

**Output**: Real-time psychological state with GUI overlay

---

## Testing & Validation

### Scenario-Based Testing

Phase 3 is tested with **psychological scenarios**, not accuracy metrics:

**Scenario 1: Happy under stress**
```
Person smiling at work but stressed about deadline
Expected: HAPPY_UNDER_STRESS, risk=HIGH
```

**Scenario 2: Emotional masking**
```
Person maintaining neutral face but voice shows fear
Expected: EMOTIONALLY_MASKED, hidden=fear
```

**Scenario 3: Emotional instability**
```
Rapid mood swings (happy→sad→angry→neutral)
Expected: EMOTIONALLY_UNSTABLE, stability<30%
```

**Scenario 4: Calm and stable**
```
Consistent neutral emotion, low stress, no switches
Expected: CALM, risk=LOW, stability>90%
```

### Run Tests

```bash
python inference/phase3_multimodal_fusion.py
```

This runs all 4 scenarios and prints detailed analysis.

---

## Phase 3 Completion Criteria

✅ **Phase 3 is COMPLETE when:**

1. Face + voice contradictions handled correctly
2. Stress influences final emotion appropriately
3. Temporal stability computed accurately
4. Outputs are explainable (not black box)
5. Mental states feel human-valid
6. Risk assessment makes psychological sense

---

## Future Enhancements

### Short-term
- [ ] Add more mental states (e.g., `EXCITED`, `CONTEMPLATIVE`)
- [ ] Tune fusion rule weights with user feedback
- [ ] Add confidence calibration with real-world data

### Medium-term
- [ ] Machine learning fusion (replace rule-based with learned weights)
- [ ] Context awareness (time of day, activity, history)
- [ ] Multi-person tracking (group dynamics)

### Long-term
- [ ] Personality trait inference (Big 5)
- [ ] Long-term mental health monitoring
- [ ] Therapeutic intervention suggestions

---

## File Structure

```
inference/
├── phase3_multimodal_fusion.py      # Phase 3 core system
├── integrated_psychologist_ai.py     # All phases integrated
├── dual_model_emotion_detection.py   # Phase 1 (face)
└── microphone_emotion_detection.py   # Phase 2 (voice)
```

---

## API Reference

### Main Class: `Phase3MultiModalFusion`

**Methods:**

```python
def process_frame(
    face_emotion: str,           # 'happy', 'sad', etc.
    face_confidence: float,      # 0.0 to 1.0
    face_detected: bool,         # True if face found
    voice_emotion: str,          # 'happy', 'sad', etc.
    voice_confidence: float,     # 0.0 to 1.0
    audio_quality: float,        # 0.0 to 1.0
    stress_level: str,           # 'low', 'medium', 'high'
    stress_confidence: float     # 0.0 to 1.0
) -> PsychologicalState
```

**Returns:** `PsychologicalState` object with:
- `dominant_emotion`: Primary emotion
- `hidden_emotion`: Secondary/masked emotion (or None)
- `mental_state`: Psychological state enum
- `confidence`: Overall confidence (0-1)
- `explanations`: List of reasoning strings
- `risk_level`: Risk assessment enum
- `stability_score`: Emotional stability (0-1)
- `temporal_patterns`: Detected patterns
- `raw_signals`: Original normalized signals
- `timestamp`: Processing time

---

## Contributing

Phase 3 is **rule-based** by design for interpretability. When adding new rules:

1. Add psychological justification (cite research if possible)
2. Test with scenarios, not just metrics
3. Ensure explanations are human-readable
4. Consider edge cases (conflicting signals)
5. Validate with real users if possible

---

## References

**Psychological principles used:**
- Emotional masking (social psychology)
- Stress dominance (stress psychology)
- Voice reliability > face (deception detection)
- Temporal stability (affective science)

**Related work:**
- Multi-modal emotion recognition
- Affective computing
- Stress detection from speech
- Facial action coding system (FACS)

---

## Contact

For questions about Phase 3 design or implementation:
- Technical: Check code comments in `phase3_multimodal_fusion.py`
- Psychological: See reasoning in Layer 4 documentation above
- Integration: Check `integrated_psychologist_ai.py` for full system

---

**Phase 3 is now COMPLETE! 🎉**

You have a fully functional psychological reasoning system that goes beyond simple emotion detection.
