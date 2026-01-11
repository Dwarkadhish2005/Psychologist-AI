# 🚀 PHASE 5: QUICK START GUIDE

**Get personality inference running in 5 minutes**

---

## ⚡ Quick Setup

### **1. Import Phase 5**

```python
from inference.phase5_personality_engine import PersonalityEngine
```

### **2. Initialize Engine**

```python
engine = PersonalityEngine(
    user_id="your_user_id",
    storage_dir="data/user_memory",
    learning_rate=0.03,          # Slow updates (recommended)
    min_sessions_required=3       # Minimum sessions before PSV
)
```

### **3. Integrate with Phase 4**

Phase 5 is **automatically integrated** in Phase 4:

```python
from inference.phase4_cognitive_layer import Phase4CognitiveFusion

# Phase 5 is already inside!
phase4 = Phase4CognitiveFusion(user_id="john_doe")

# After 3+ sessions, get PSV
psv_summary = phase4.get_phase5_personality_summary()
print(psv_summary)
```

---

## 📊 PSV Traits (The 5 Core Dimensions)

| Trait | Range | Interpretation |
|-------|-------|----------------|
| **Emotional Stability** | 0-1 | How consistent emotions are |
| **Stress Sensitivity** | 0-1 | How easily stress rises |
| **Recovery Speed** | 0-1 | How fast return to baseline |
| **Positivity Bias** | 0-1 | Emotional orientation |
| **Volatility** | 0-1 | State transition frequency |

**All values:**
- **0.0-0.4:** Low
- **0.4-0.7:** Moderate
- **0.7-1.0:** High

---

## 🎯 Common Tasks

### **Check if PSV is Ready**

```python
if engine.can_infer_personality():
    print("✅ PSV active")
else:
    print(f"⏳ Need {engine.min_sessions_required} sessions")
```

### **Get PSV Summary**

```python
summary = engine.get_personality_summary()

psv = summary['personality_state_vector']
print(f"Emotional Stability: {psv['emotional_stability']:.3f}")
print(f"Stress Sensitivity: {psv['stress_sensitivity']:.3f}")
print(f"Recovery Speed: {psv['recovery_speed']:.3f}")
print(f"Positivity Bias: {psv['positivity_bias']:.3f}")
print(f"Volatility: {psv['volatility']:.3f}")
```

### **Generate Formatted Report**

```python
from inference.phase5_personality_engine import generate_personality_report

report = generate_personality_report(engine)
print(report)
```

### **Export for Research**

```python
research_data = engine.export_psv_for_research()

import json
with open('psv_export.json', 'w') as f:
    json.dump(research_data, f, indent=2)
```

---

## 🔧 Configuration

### **Learning Rate (η)**

Controls how fast PSV updates:

```python
engine = PersonalityEngine(
    user_id="john",
    learning_rate=0.01  # Very slow (research)
    # learning_rate=0.03  # Default (balanced)
    # learning_rate=0.05  # Moderate (quick adaptation)
)
```

### **Decay Lambda (λ)**

Controls temporal weighting:

```python
engine = PersonalityEngine(
    user_id="john",
    decay_lambda=0.1   # Default (recent > old)
    # decay_lambda=0.05  # Slower decay (history matters more)
    # decay_lambda=0.2   # Faster decay (recent matters most)
)
```

### **Min Sessions**

When to start PSV inference:

```python
engine = PersonalityEngine(
    user_id="john",
    min_sessions_required=3   # Default (3+ sessions)
    # min_sessions_required=5   # Conservative (5+ sessions)
    # min_sessions_required=1   # Aggressive (1+ session)
)
```

---

## 📖 Output Formats

### **Summary Dictionary**

```python
{
    "personality_state_vector": {
        "emotional_stability": 0.682,
        "stress_sensitivity": 0.594,
        "recovery_speed": 0.721,
        "positivity_bias": 0.558,
        "volatility": 0.412
    },
    "trends": {
        "emotional_stability": "stable",
        "stress_sensitivity": "increasing",
        "recovery_speed": "stable",
        "positivity_bias": "decreasing",
        "volatility": "stable"
    },
    "confidence": {
        "level": "high",
        "score": 0.78,
        "sessions_processed": 39
    },
    "behavioral_descriptor": "User demonstrates consistent emotional patterns...",
    "last_updated": "2026-01-11T14:30:00",
    "ethical_notice": "This is a behavioral pattern assessment, not a clinical diagnosis."
}
```

### **Formatted Report (Text)**

```
╔══════════════════════════════════════════════════════════════════════╗
║            PERSONALITY PATTERN ASSESSMENT REPORT                     ║
╚══════════════════════════════════════════════════════════════════════╝

User ID: john_doe
Generated: 2026-01-11 14:30:00

─────────────────────────────────────────────────────────────────────

PERSONALITY STATE VECTOR (PSV)

  • Emotional Stability:    0.682  [stable]
  • Stress Sensitivity:     0.594  [increasing]
  • Recovery Speed:         0.721  [stable]
  • Positivity Bias:        0.558  [decreasing]
  • Volatility:             0.412  [stable]

...
```

---

## 🛡️ Safety Checklist

Before deploying Phase 5:

- [ ] Minimum 3 sessions before PSV updates
- [ ] Learning rate ≤ 0.05 (prevents noise)
- [ ] Ethical notice included in all outputs
- [ ] No diagnostic language in reports
- [ ] Confidence levels displayed
- [ ] Research/demo context clarified

---

## 🔥 Integration Patterns

### **Pattern 1: Automatic (Recommended)**

```python
# Phase 5 runs automatically inside Phase 4
phase4 = Phase4CognitiveFusion(user_id="john")

# Just use the system normally
for state in psychological_states:
    profile = phase4.process_state(state)

# PSV updates automatically after each session
```

### **Pattern 2: Manual Control**

```python
# Direct Phase 5 access
from inference.phase5_personality_engine import PersonalityEngine
from inference.phase4_cognitive_layer import LongTermMemory

engine = PersonalityEngine(user_id="john")
memory = LongTermMemory(user_id="john")

# Update PSV manually
recent_profiles = memory.get_recent_daily_profiles(7)
engine.update_psv(recent_profiles)

# Get results
psv_summary = engine.get_personality_summary()
```

---

## 💡 Pro Tips

1. **Wait for 3+ sessions** - PSV needs data to stabilize
2. **Use η=0.03** - Default learning rate is well-tuned
3. **Monitor trends** - More important than absolute values
4. **Include confidence** - Always show confidence level
5. **No diagnoses** - Keep language behavioral, not clinical
6. **Export regularly** - Save PSV data for longitudinal analysis

---

## 🚨 Troubleshooting

### **Problem: PSV not updating**

```python
# Check if enough sessions
if not engine.can_infer_personality():
    print(f"Current: {engine.psv.total_sessions_processed}")
    print(f"Required: {engine.min_sessions_required}")
```

### **Problem: PSV values unstable**

```python
# Reduce learning rate
engine.learning_rate = 0.01  # Slower updates

# Or increase min sessions
engine.min_sessions_required = 5  # More data before inference
```

### **Problem: Old data dominating**

```python
# Increase decay lambda
engine.decay_lambda = 0.2  # Faster decay (recent > old)
```

---

## 📚 Next Steps

- Read [PHASE5_COMPLETE.md](PHASE5_COMPLETE.md) for full documentation
- See [STORAGE_DETAILS.md](STORAGE_DETAILS.md) for PSV storage info
- Check [phase5_personality_engine.py](../inference/phase5_personality_engine.py) for source code

---

**Ready to infer personality? Let's go! 🚀**

**Last Updated:** January 11, 2026
