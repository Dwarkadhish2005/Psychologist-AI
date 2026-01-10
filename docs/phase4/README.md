# 🧠 PHASE 4: LONG-TERM COGNITIVE LAYER

## Overview

**Phase 4** transforms the Psychologist AI from momentary state detection to **personalized psychological understanding**. It adds memory, personality inference, and behavioral baseline tracking to answer the critical question: **"Is this behavior unusual FOR THIS PERSON?"**

---

## 🎯 What Phase 4 Solves

### Problems with Phase 3 Alone:
- ❌ No memory between sessions
- ❌ Treats all users the same
- ❌ Can't distinguish "stressed person" from "stress-prone personality"
- ❌ No concept of "normal for this individual"
- ❌ Risk thresholds are absolute, not personalized

### Phase 4 Solutions:
- ✅ Cross-session memory (days, weeks, months)
- ✅ Personality trait inference
- ✅ Personal baseline establishment
- ✅ Deviation detection (unusual for THIS person)
- ✅ Personalized risk assessment

---

## 🧩 Architecture

```
PHASE 3 (PsychologicalState)
        ↓
┌─────────────────────────────────────┐
│      PHASE 4: COGNITIVE LAYER       │
├─────────────────────────────────────┤
│                                     │
│  1. SessionMemory                   │
│     └─ 30-90 min tracking           │
│                                     │
│  2. LongTermMemory                  │
│     └─ Daily/Weekly summaries       │
│                                     │
│  3. PersonalityProfile              │
│     └─ Trait inference              │
│                                     │
│  4. BaselineProfile                 │
│     └─ Personal "normal"            │
│                                     │
│  5. DeviationDetector               │
│     └─ Anomaly detection            │
│                                     │
│  6. UserPsychologicalProfile        │
│     └─ Complete output              │
│                                     │
└─────────────────────────────────────┘
        ↓
UserPsychologicalProfile
```

---

## 📊 Core Concepts

### 1. **STATE ≠ TRAIT**

| Type | Example | Duration |
|------|---------|----------|
| **State** | Stressed right now | Minutes-Hours |
| **Trait** | Stress-prone personality | Stable over time |

**Phase 4 converts states → traits** through statistical aggregation.

### 2. **BASELINE > ABSOLUTE**

**Example:**
- **Person A**: Usually calm, stress = 30% baseline
- **Person B**: Often stressed, stress = 60% baseline

**Today both at 50% stress:**
- Person A: 🚨 **DEVIATION** (20% above baseline)
- Person B: ✅ **NORMAL** (10% below baseline)

**Phase 4 compares current vs personal baseline**, not absolute thresholds.

### 3. **DEVIATION DETECTION**

Phase 4 detects 5 anomaly types:
1. **Sudden stress spike** - Stress much higher than normal
2. **Prolonged instability** - Emotional stability dropped
3. **Unusual masking** - Hiding emotions more than usual
4. **Mood polarity shift** - Dramatic mood change
5. **Risk escalation** - Risk jumped significantly

---

## 🧱 7 Modules

### **MODULE 1: SessionMemory**
**Purpose:** Track single session (30-90 mins)

**Stores:**
- PsychologicalState objects with timestamps
- 20+ session metrics

**Outputs:**
- `SessionMetrics` - Comprehensive session statistics

**Key Metrics:**
- Dominant mental states
- Stress duration %
- Masking frequency
- Risk escalations
- State switches

---

### **MODULE 2: LongTermMemory**
**Purpose:** Store cross-session data (days/weeks)

**Stores:**
- Daily profiles (DailyProfile)
- Weekly aggregates (WeeklyAggregate)
- Trend slopes

**Persistence:**
- JSON file storage
- Auto-loads on init
- Survives application restarts

**Provides:**
- `get_recent_days(n)` - Last N days
- `get_recent_weeks(n)` - Last N weeks
- `get_overall_trends()` - Long-term slopes

---

### **MODULE 3: PersonalityProfile**
**Purpose:** Infer personality traits

**5 Core Traits:**

| Trait | Meaning | Scale |
|-------|---------|-------|
| **Emotional Reactivity** | How fast emotions change | 0=stable, 1=reactive |
| **Stress Tolerance** | Handles stress before risk rises | 0=sensitive, 1=tolerant |
| **Masking Tendency** | Hides emotions or not | 0=authentic, 1=masks |
| **Emotional Stability** | Variance over time | 0=unstable, 1=stable |
| **Baseline Mood** | Typical emotional tone | positive/neutral/negative |

**Inference Method:**
- Pure statistics (no ML)
- Aggregates behavioral patterns
- Minimum 3 days for reliability
- Confidence score included

**Example:**
```python
PersonalityProfile(
    emotional_reactivity=0.35,  # Calm
    stress_tolerance=0.72,      # Good tolerance
    masking_tendency=0.18,      # Authentic
    emotional_stability=0.81,   # Very stable
    baseline_mood="positive",
    confidence=0.85,            # 85% confident
    data_days=14
)
```

---

### **MODULE 4: BaselineProfile**
**Purpose:** Define "normal for this user"

**Contains:**
- Average stress level
- Average stability
- Typical mental states
- Normal risk level
- **Deviation thresholds** (mean + 1.5σ)

**Uses:**
- Last 7-30 days of data
- Statistical averaging
- Adaptive thresholds

**Example:**
```python
BaselineProfile(
    avg_stress_level=0.32,      # Usually 32% stressed
    avg_stability=0.75,         # Usually 75% stable
    stress_threshold=0.58,      # Alert if > 58%
    stability_threshold=0.55,   # Alert if < 55%
    confidence=0.78
)
```

---

### **MODULE 5: DeviationDetector**
**Purpose:** Detect behavioral anomalies

**Detection Logic:**
```python
if current_value > baseline_value + threshold:
    deviation = True
```

**5 Deviation Types:**

| Type | Trigger | Severity |
|------|---------|----------|
| `sudden_stress_spike` | Stress >> normal | (current - threshold) / range |
| `prolonged_instability` | Stability << normal | (threshold - current) / threshold |
| `unusual_masking` | Masking >> normal | (current - threshold) / threshold |
| `mood_polarity_shift` | Positive ↔ Negative | Polarity ratio |
| `risk_escalation` | Risk increased > 1 level | Risk delta / 2 |

**Output:**
```python
Deviation(
    deviation_type="sudden_stress_spike",
    severity=0.82,  # 82% severe
    description="Stress level (65%) significantly higher than normal (32%)",
    current_value=0.65,
    baseline_value=0.32,
    threshold=0.58
)
```

---

### **MODULE 6: UserPsychologicalProfile**
**Purpose:** Complete Phase 4 output

**Contains:**
```python
UserPsychologicalProfile(
    personality: PersonalityProfile,
    baseline: BaselineProfile,
    current_state: PsychologicalState,  # From Phase 3
    current_session_metrics: SessionMetrics,
    deviations: List[Deviation],
    phase3_risk: RiskLevel,             # Original
    adjusted_risk: RiskLevel,           # Personalized
    risk_adjustment_reason: str,
    confidence: float
)
```

**This is the FINAL system output** - combines all Phase 3 + Phase 4 intelligence.

---

### **MODULE 7: Phase4CognitiveFusion**
**Purpose:** Orchestrate entire Phase 4 system

**Main Method:**
```python
profile = phase4.process_state(psychological_state)
```

**Workflow:**
1. Add state to session memory
2. Check session timeout
3. Update long-term memory if session ended
4. Refresh personality/baseline (daily)
5. Detect deviations
6. Adjust risk
7. Return complete profile

---

## 🔥 Personalized Risk Assessment

**This is Phase 4's killer feature.**

### Risk Adjustment Logic:

```python
adjusted_risk = phase3_risk + adjustments

Adjustments:
  +1 if: Severe deviations detected
  +1 if: Low resilience + high volatility personality
  +1 if: Sudden stress spike OR prolonged instability
  -1 if: High resilience + no deviations
  -1 if: High baseline stress (stress is normal for user)
```

### Examples:

**Example 1: Elevation**
```
User: Usually calm (stress_baseline=0.2)
Current: Very stressed (stress=0.75)
Deviation: sudden_stress_spike (severity=0.9)

Phase 3 Risk: MODERATE
Phase 4 Risk: HIGH ⬆️
Reason: "Risk elevated: 1 severe deviation, sudden_stress_spike"
```

**Example 2: Reduction**
```
User: Stress-prone (stress_baseline=0.65, resilience=0.8)
Current: Moderately stressed (stress=0.55)
Deviation: None

Phase 3 Risk: MODERATE
Phase 4 Risk: LOW ⬇️
Reason: "Risk reduced: stress is normal for this user, high resilience"
```

---

## 📈 Data Flow

```
┌────────────────────┐
│   Real-time Frame  │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│   Phase 1 + 2 + 3  │ → PsychologicalState
└─────────┬──────────┘
          ↓
┌────────────────────┐
│  SessionMemory     │ → Add state
│  (30-90 mins)      │
└─────────┬──────────┘
          ↓
    Session ends?
          ↓ YES
┌────────────────────┐
│  LongTermMemory    │ → Save DailyProfile
│  (persistent JSON) │
└─────────┬──────────┘
          ↓
   24 hours passed?
          ↓ YES
┌────────────────────┐
│  Personality +     │ → Update profiles
│  Baseline Rebuild  │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Deviation Detector │ → Check anomalies
└─────────┬──────────┘
          ↓
┌────────────────────┐
│  Risk Adjuster     │ → Personalize risk
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ UserPsychological  │ → Output
│      Profile       │
└────────────────────┘
```

---

## 🚀 Usage

### Basic Integration:

```python
from inference.phase4_cognitive_layer import Phase4CognitiveFusion
from inference.phase3_multimodal_fusion import Phase3MultiModalFusion

# Initialize
phase3 = Phase3MultiModalFusion()
phase4 = Phase4CognitiveFusion(user_id="john_doe")

# Process frame
psychological_state = phase3.process_frame(...)  # Phase 3 output
profile = phase4.process_state(psychological_state)  # Phase 4 output

# Access results
print(f"Current emotion: {profile.current_state.dominant_emotion}")
print(f"Personality: {profile.personality.baseline_mood}")
print(f"Deviations: {len(profile.deviations)}")
print(f"Risk (Phase 3): {profile.phase3_risk.value}")
print(f"Risk (Phase 4): {profile.adjusted_risk.value}")
```

### Full Report:

```python
report = phase4.get_full_report(profile)
print(report)
```

Output:
```
================================================================================
🧠 COMPLETE PSYCHOLOGICAL PROFILE
================================================================================

👤 USER: john_doe
📅 TIMESTAMP: 2026-01-10T20:15:32
🎯 CONFIDENCE: 78%

────────────────────────────────────────────────────────────────────────────────
📊 CURRENT STATE (PHASE 3)
────────────────────────────────────────────────────────────────────────────────
Dominant Emotion: anxious
Mental State: stressed
Risk Level: moderate

────────────────────────────────────────────────────────────────────────────────
🎭 PERSONALITY TRAITS (LONG-TERM)
────────────────────────────────────────────────────────────────────────────────
Emotional Reactivity: 0.42 (stable)
Stress Tolerance: 0.68 (good)
Emotional Stability: 0.75 (stable)
Baseline Mood: positive

────────────────────────────────────────────────────────────────────────────────
🚨 DEVIATION ANALYSIS
────────────────────────────────────────────────────────────────────────────────
⚠️ 2 behavioral deviation(s) detected:
  1. Stress level (58%) significantly higher than normal (32%)
  2. Hiding emotions (4.2/min) much more than usual (1.8/min)

────────────────────────────────────────────────────────────────────────────────
🎯 PERSONALIZED RISK ASSESSMENT
────────────────────────────────────────────────────────────────────────────────
Phase 3 Risk: moderate
Adjusted Risk: high
Adjustment: ⬆️ Risk elevated: 2 severe deviations, sudden_stress_spike
================================================================================
```

---

## 📁 Files

| File | Purpose | Lines |
|------|---------|-------|
| [phase4_cognitive_layer.py](../../inference/phase4_cognitive_layer.py) | Complete Phase 4 implementation | 2000+ |
| [integrated_psychologist_ai.py](../../inference/integrated_psychologist_ai.py) | Integration with Phase 3 | 600+ |
| [demo_phase4_integration.py](../../inference/demo_phase4_integration.py) | Interactive demo | 200+ |

---

## 🧪 Testing

### Run Phase 4 Tests:
```bash
python inference/phase4_cognitive_layer.py
```

### Run Integration Demo:
```bash
python inference/demo_phase4_integration.py
```

### Run Full System:
```bash
python inference/integrated_psychologist_ai.py
```

---

## 🎓 Key Insights

### 1. **No Machine Learning Required**
Phase 4 uses pure statistics:
- Averages
- Standard deviations
- Linear regression (trends)
- Thresholds (mean + 1.5σ)

**Why?** Explainability. Every decision can be explained.

### 2. **Minimum Data Requirements**
- **1 day**: Limited personality inference
- **3 days**: Reliable personality (70% confidence)
- **7 days**: Strong baseline (80% confidence)
- **14+ days**: High confidence (90%+)

### 3. **Memory Efficiency**
- SessionMemory: RAM only
- LongTermMemory: JSON file (~50KB per month)
- No database required

### 4. **Privacy by Design**
- Data stored locally
- User-specific files
- No cloud uploads
- Can be deleted anytime

---

## 🔮 Future Enhancements

### Planned Features:
1. **Multi-user support** - Compare users
2. **Intervention triggers** - Auto-suggest when to act
3. **Trend predictions** - Forecast future states
4. **Social context** - Time of day, day of week patterns
5. **Export reports** - PDF generation

---

## 📚 Related Documentation

- [Phase 3 Documentation](../phase3/PHASE3_DOCUMENTATION.md)
- [Integration Guide](../../README.md)
- [API Reference](API_REFERENCE.md) *(coming soon)*

---

## 🙏 Credits

**Phase 4 Design Philosophy:**
> "Understanding someone requires knowing their history, not just their present."

Built with ❤️ by the Psychologist AI Team

---

**Last Updated:** January 10, 2026
**Version:** 1.0.0
**Status:** ✅ Production Ready
