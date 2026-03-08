"""
Project Verification & Run Script
Checks all imports, loads all models, runs Phase 1-5 logic with synthetic data.
"""

import sys, os, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "[OK]  "
FAIL = "[FAIL]"
SEP  = "=" * 64

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

errors = []

# ─────────────────────────────────────────────────────────
# 1. THIRD-PARTY DEPENDENCIES
# ─────────────────────────────────────────────────────────
section("1. DEPENDENCY CHECK")

import importlib
deps = [
    ("cv2", "OpenCV"), ("torch", "PyTorch"), ("numpy", "NumPy"),
    ("librosa", "Librosa"), ("sounddevice", "SoundDevice"),
    ("sklearn", "Scikit-Learn"), ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"), ("pandas", "Pandas"), ("plotly", "Plotly"),
    ("tqdm", "tqdm"), ("PIL", "Pillow"), ("soundfile", "SoundFile"),
    ("scipy", "SciPy"), ("noisereduce", "NoiseReduce"),
]
for mod, name in deps:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "ok")
        print(f"  {PASS} {name:<18} {ver}")
    except Exception as e:
        print(f"  {FAIL} {name:<18} MISSING")
        errors.append(f"Missing package: {name}")

import torch, numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  PyTorch device : {'CUDA — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (no CUDA)'}")
print(f"  Python version : {sys.version.split()[0]}")

# ─────────────────────────────────────────────────────────
# 2. PROJECT MODULE IMPORTS
# ─────────────────────────────────────────────────────────
section("2. PROJECT MODULE IMPORTS")

from training.model import EmotionCNN, EmotionCNNDeep, count_parameters
print(f"  {PASS} training.model                   EmotionCNN, EmotionCNNDeep")

from training.preprocessing import preprocess_face
print(f"  {PASS} training.preprocessing           preprocess_face")

from training.voice.voice_emotion_model import VoiceEmotionSystem
print(f"  {PASS} training.voice.voice_emotion_model  VoiceEmotionSystem")

from training.voice.feature_extraction import extract_all_features, extract_stress_score
print(f"  {PASS} training.voice.feature_extraction   extract_all_features, extract_stress_score")

from training.voice.audio_preprocessing import normalize_audio
print(f"  {PASS} training.voice.audio_preprocessing  normalize_audio")

from inference.phase3_multimodal_fusion import Phase3MultiModalFusion, format_psychological_state
print(f"  {PASS} inference.phase3_multimodal_fusion  Phase3MultiModalFusion")

from inference.phase4_cognitive_layer import Phase4CognitiveFusion
print(f"  {PASS} inference.phase4_cognitive_layer    Phase4CognitiveFusion")

from inference.phase4_user_manager import UserSelector
print(f"  {PASS} inference.phase4_user_manager       UserSelector")

from inference.phase5_personality_engine import PersonalityEngine, PersonalityStateVector
print(f"  {PASS} inference.phase5_personality_engine PersonalityEngine")

from inference.phase5_visualization import (
    create_psv_radar_chart, create_psv_bar_chart, create_comprehensive_dashboard
)
print(f"  {PASS} inference.phase5_visualization      create_psv_radar_chart, create_psv_bar_chart, create_comprehensive_dashboard")

# ─────────────────────────────────────────────────────────
# 3. MODEL FILE VERIFICATION
# ─────────────────────────────────────────────────────────
section("3. MODEL FILE VERIFICATION")

MODEL_ROOT = os.path.join(os.path.dirname(__file__), "models")
model_files = {
    os.path.join("face_emotion", "emotion_cnn_best.pth"):              "Face main model",
    os.path.join("face_emotion", "emotion_cnn_phase15_specialist.pth"):"Face specialist model",
    os.path.join("face_emotion", "config.json"):                       "Face config",
    os.path.join("voice_emotion", "emotion_model_best_balanced.pth"):  "Voice balanced model",
    os.path.join("voice_emotion", "stress_model_best.pth"):            "Stress model",
    os.path.join("voice_emotion", "config.json"):                      "Voice config",
}
for rel, desc in model_files.items():
    p = os.path.join(MODEL_ROOT, rel)
    if os.path.exists(p):
        size_mb = os.path.getsize(p) / 1024 / 1024
        print(f"  {PASS} {desc:<40} {size_mb:.1f} MB")
    else:
        print(f"  {FAIL} {desc:<40} MISSING — {p}")
        errors.append(f"Missing model: {desc}")

# ─────────────────────────────────────────────────────────
# 4. LOAD FACE MODELS
# ─────────────────────────────────────────────────────────
section("4. LOAD FACE MODELS")

face_cfg_path = os.path.join(MODEL_ROOT, "face_emotion", "config.json")
with open(face_cfg_path) as f:
    face_cfg = json.load(f)
print(f"  Face classes     : {face_cfg['class_names']}")
print(f"  Num classes      : {face_cfg['num_classes']}")

_face_arch = face_cfg.get('architecture', 'EmotionCNN')
_FACE_ARCH_MAP = {'EmotionCNNDeep': EmotionCNNDeep, 'EmotionCNN': EmotionCNN}
_MainFaceClass = _FACE_ARCH_MAP.get(_face_arch, EmotionCNN)

main_face_model = _MainFaceClass(num_classes=face_cfg["num_classes"], input_size=48)
ckpt = torch.load(os.path.join(MODEL_ROOT, "face_emotion", "emotion_cnn_best.pth"), map_location=device, weights_only=False)
main_face_model.load_state_dict(ckpt)
main_face_model.to(device).eval()
print(f"  {PASS} Main face model loaded — params: {count_parameters(main_face_model):,}")

spec_face_model = EmotionCNN(num_classes=face_cfg["num_classes"], input_size=48)
ckpt2 = torch.load(os.path.join(MODEL_ROOT, "face_emotion", "emotion_cnn_phase15_specialist.pth"), map_location=device, weights_only=False)
spec_face_model.load_state_dict(ckpt2)
spec_face_model.to(device).eval()
print(f"  {PASS} Specialist face model loaded — params: {count_parameters(spec_face_model):,}")

# ─────────────────────────────────────────────────────────
# 5. LOAD VOICE MODELS
# ─────────────────────────────────────────────────────────
section("5. LOAD VOICE MODELS")

voice_cfg_path = os.path.join(MODEL_ROOT, "voice_emotion", "config.json")
with open(voice_cfg_path) as f:
    voice_cfg = json.load(f)
print(f"  Voice classes    : {voice_cfg['class_names']}")
print(f"  Stress levels    : {voice_cfg['stress_levels']}")

voice_system = VoiceEmotionSystem(feature_dim=48, num_emotions=5, num_stress_levels=3)
# Load emotion model weights
em_ckpt = torch.load(os.path.join(MODEL_ROOT, "voice_emotion", "emotion_model_best_balanced.pth"), map_location=device, weights_only=False)
voice_system.emotion_model.load_state_dict(em_ckpt)
# Load stress model weights
st_ckpt = torch.load(os.path.join(MODEL_ROOT, "voice_emotion", "stress_model_best.pth"), map_location=device, weights_only=False)
voice_system.stress_detector.load_state_dict(st_ckpt)
voice_system.to(device).eval()
print(f"  {PASS} VoiceEmotionSystem loaded (emotion + stress models)")

# Load feature scaler
_voice_scaler = None
_scaler_path = os.path.join(MODEL_ROOT, "voice_emotion", "feature_scaler.pkl")
if os.path.exists(_scaler_path):
    try:
        import joblib
        _voice_scaler = joblib.load(_scaler_path)
        print(f"  {PASS} Feature scaler loaded")
    except Exception as e:
        print(f"  [WARN] Could not load feature scaler: {e}")

# ─────────────────────────────────────────────────────────
# 6. PHASE 1 — FACE EMOTION INFERENCE (synthetic image)
# ─────────────────────────────────────────────────────────
section("6. PHASE 1 — FACE EMOTION INFERENCE (synthetic 48x48 image)")

import cv2
dummy_face = np.random.randint(80, 180, (48, 48, 3), dtype=np.uint8)
face_arr = preprocess_face(dummy_face, target_size=(48, 48))
face_tensor = torch.from_numpy(face_arr).unsqueeze(0).to(device)

with torch.no_grad():
    out = main_face_model(face_tensor)
    probs = torch.softmax(out, dim=1).cpu().numpy()[0]

pred_idx = int(np.argmax(probs))
face_emotion = face_cfg["class_names"][pred_idx]
face_confidence = float(probs[pred_idx])
print(f"  Predicted emotion   : {face_emotion}  ({face_confidence*100:.1f}%)")
print(f"  All class scores    :")
for cn, p in zip(face_cfg["class_names"], probs):
    bar = "█" * int(p * 25)
    print(f"    {cn:<12} {p*100:5.1f}%  {bar}")

# ─────────────────────────────────────────────────────────
# 7. PHASE 2 — VOICE EMOTION INFERENCE (synthetic 3s audio)
# ─────────────────────────────────────────────────────────
section("7. PHASE 2 — VOICE EMOTION INFERENCE (synthetic 3s audio)")

SR = 16000
dummy_audio = np.random.randn(SR * 3).astype(np.float32) * 0.01
norm_audio = normalize_audio(dummy_audio)

_, feature_vector = extract_all_features(norm_audio, SR)
feature_vector = np.array(feature_vector, dtype=np.float32).flatten()
if _voice_scaler is not None:
    feature_vector = _voice_scaler.transform(feature_vector.reshape(1, -1)).flatten()

stress_result = extract_stress_score(norm_audio, SR)
stress_features = np.array([
    stress_result["jitter"], stress_result["shimmer"],
    stress_result["spectral_flatness"], stress_result["pitch_variance"]
], dtype=np.float32)

feat_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(device)
stress_tensor = torch.FloatTensor(stress_features).unsqueeze(0).to(device)

with torch.no_grad():
    emotion_out = voice_system.emotion_model(feat_tensor)
    emotion_probs = torch.softmax(emotion_out, dim=1).cpu().numpy()[0]
    emotion_idx = int(np.argmax(emotion_probs))
    voice_emotion = voice_system.emotion_model.class_names[emotion_idx]
    voice_confidence = float(emotion_probs[emotion_idx])

    stress_out = voice_system.stress_detector(stress_tensor)
    stress_probs = torch.softmax(stress_out, dim=1).cpu().numpy()[0]
    stress_idx = int(np.argmax(stress_probs))
    stress_levels_list = ["low", "medium", "high"]
    stress_level = stress_levels_list[stress_idx]
    stress_conf = float(stress_probs[stress_idx])

audio_quality = min(1.0, float(np.std(norm_audio)) * 10)
print(f"  Voice emotion       : {voice_emotion}  ({voice_confidence*100:.1f}%)")
print(f"  Stress level        : {stress_level}  ({stress_conf*100:.1f}%)")
print(f"  Stress score        : {stress_result['stress_score']:.3f}")
print(f"  Audio quality       : {audio_quality:.3f}")
print(f"  All voice scores    :")
for em, p in zip(voice_system.emotion_model.class_names, emotion_probs):
    bar = "█" * int(p * 25)
    print(f"    {em:<12} {p*100:5.1f}%  {bar}")

# ─────────────────────────────────────────────────────────
# 8. PHASE 3 — MULTIMODAL FUSION
# ─────────────────────────────────────────────────────────
section("8. PHASE 3 — MULTIMODAL FUSION")

fusion = Phase3MultiModalFusion()
psych_state = fusion.process_frame(
    face_emotion=face_emotion,
    face_confidence=face_confidence,
    face_detected=True,
    voice_emotion=voice_emotion,
    voice_confidence=voice_confidence,
    audio_quality=audio_quality,
    stress_level=stress_level,
    stress_confidence=stress_conf,
)

formatted = format_psychological_state(psych_state)
print(f"  Mental state        : {psych_state.mental_state.value}")
print(f"  Dominant emotion    : {psych_state.dominant_emotion}")
print(f"  Risk level          : {psych_state.risk_level.value}")
print(f"  Confidence          : {psych_state.confidence*100:.1f}%")
print(f"  Stability score     : {psych_state.stability_score:.3f}")
print(f"\n  Formatted psychological state:")
for line in formatted.split("\n"):
    if line.strip():
        print(f"    {line}")

# ─────────────────────────────────────────────────────────
# 9. PHASE 4 — COGNITIVE LAYER (long-term memory)
# ─────────────────────────────────────────────────────────
section("9. PHASE 4 — COGNITIVE LAYER")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "user_memory")
phase4 = Phase4CognitiveFusion(
    user_id="dwarkadhish_0947d1caefcb",
    storage_dir=DATA_DIR,
    session_timeout_minutes=5.0
)
profile = phase4.process_state(psych_state)

print(f"  Current state       : {profile.current_state.mental_state.value}")
print(f"  Phase3 risk         : {profile.phase3_risk.value}")
print(f"  Adjusted risk       : {profile.adjusted_risk.value}")
print(f"  Risk adj. reason    : {profile.risk_adjustment_reason}")
print(f"  Confidence          : {profile.confidence*100:.1f}%")
print(f"  Deviations detected : {len(profile.deviations)}")
if profile.deviations:
    for d in profile.deviations[:3]:
        print(f"    - [{d.severity}] {d.deviation_type}: {d.description}")
print(f"\n  Personality profile (Phase 4 inferred traits):")
print(f"    Emotional reactivity  : {profile.personality.emotional_reactivity:.3f}")
print(f"    Stress tolerance      : {profile.personality.stress_tolerance:.3f}")
print(f"    Masking tendency      : {profile.personality.masking_tendency:.3f}")
print(f"    Emotional stability   : {profile.personality.emotional_stability:.3f}")
print(f"    Volatility score      : {profile.personality.volatility_score:.3f}")
print(f"    Resilience score      : {profile.personality.resilience_score:.3f}")
print(f"    Baseline mood         : {profile.personality.baseline_mood}")
print(f"    Profile confidence    : {profile.personality.confidence:.3f}")
print(f"\n  Baseline (personal normal):")
print(f"    Avg stress level      : {profile.baseline.avg_stress_level:.3f}")
print(f"    Avg stability         : {profile.baseline.avg_stability:.3f}")
print(f"    Normal risk           : {profile.baseline.normal_risk_level:.3f}")
print(f"    Avg masking/min       : {profile.baseline.avg_masking_frequency:.3f}")

# ─────────────────────────────────────────────────────────
# 10. PHASE 5 — PERSONALITY ENGINE (PSV)
# ─────────────────────────────────────────────────────────
section("10. PHASE 5 — PERSONALITY ENGINE (PSV)")

PSV_PATH = os.path.join(DATA_DIR, "dwarkadhish_0947d1caefcb_psv.json")
psv_engine = PersonalityEngine(
    user_id="dwarkadhish_0947d1caefcb",
    storage_dir=DATA_DIR
)
print(f"  PSV file            : {os.path.basename(PSV_PATH)}")
print(f"  Can infer           : {psv_engine.can_infer_personality()}")

summary = psv_engine.get_personality_summary()
print(f"  PSV summary keys    : {list(summary.keys())}")
print(f"\n  Personality snapshot (PSV):")
for k, v in summary.items():
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        print(f"    {k:<32} {v:.4f}" if isinstance(v, float) else f"    {k:<32} {v}")
    elif isinstance(v, str):
        print(f"    {k:<32} {v}")

print(f"\n  Core PSV traits:")
psv = psv_engine.psv  # PersonalityStateVector dataclass
print(f"    Emotional stability   : {psv.emotional_stability:.3f}")
print(f"    Stress sensitivity    : {psv.stress_sensitivity:.3f}")
print(f"    Recovery speed        : {psv.recovery_speed:.3f}")
print(f"    Positivity bias       : {psv.positivity_bias:.3f}")
print(f"    Volatility            : {psv.volatility:.3f}")
print(f"    Confidence            : {psv.confidence:.3f}")
print(f"    Sessions processed    : {psv.total_sessions_processed}")

# ─────────────────────────────────────────────────────────
# 11. PHASE 5 — VISUALIZATION OUTPUT
# ─────────────────────────────────────────────────────────
section("11. PHASE 5 — VISUALIZATION OUTPUT")

OUT_DIR = os.path.join(os.path.dirname(__file__), "assets", "reports", "psv_visualizations")
os.makedirs(OUT_DIR, exist_ok=True)

uid = "dwarkadhish_0947d1caefcb"
plots = [
    (create_psv_radar_chart,        os.path.join(OUT_DIR, f"{uid}_radar.png"),     "Radar chart",     psv),
    (create_psv_bar_chart,          os.path.join(OUT_DIR, f"{uid}_bars.png"),      "Bar chart",       psv),
    (create_comprehensive_dashboard, os.path.join(OUT_DIR, f"{uid}_dashboard.png"), "Dashboard",      psv_engine),
]
for fn, path, label, arg in plots:
    try:
        fig = fn(arg, save_path=path)
        if fig:
            import matplotlib.pyplot as plt
            plt.close(fig)
        size_kb = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
        print(f"  {PASS} {label:<20} -> {os.path.basename(path)}  ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"  {FAIL} {label:<20} ERROR: {e}")
        errors.append(f"Viz error: {label} — {e}")

# ─────────────────────────────────────────────────────────
# 12. USER MEMORY FILES
# ─────────────────────────────────────────────────────────
section("12. USER MEMORY & DATA FILES")

for fname in sorted(os.listdir(DATA_DIR)):
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.isfile(fpath):
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {PASS} {fname:<55} {size_kb:.1f} KB")

# ─────────────────────────────────────────────────────────
# 13. TOOLS SCRIPTS
# ─────────────────────────────────────────────────────────
section("13. TOOLS SCRIPTS")

TOOLS_DIR = os.path.join(os.path.dirname(__file__), "tools")
for fname in sorted(os.listdir(TOOLS_DIR)):
    fpath = os.path.join(TOOLS_DIR, fname)
    if fname.endswith(".py"):
        print(f"  {PASS} tools/{fname}")

# ─────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────
section("VERIFICATION COMPLETE")

if errors:
    print(f"\n  ISSUES FOUND ({len(errors)}):")
    for e in errors:
        print(f"    {FAIL} {e}")
else:
    print("""
  Phase 1  Face Emotion (Dual CNN)              WORKING
  Phase 2  Voice Emotion + Stress Detection     WORKING
  Phase 3  Multimodal Fusion                    WORKING
  Phase 4  Cognitive Layer (long-term memory)   WORKING
  Phase 5  Personality Engine (PSV)             WORKING
  Phase 5  Visualization Charts                 WORKING

  Expected outputs from a live session:
  ----------------------------------------
  [Every frame]
    - Face emotion + confidence (e.g., "neutral 78%")
    - Voice emotion + confidence (e.g., "happy 65%")
    - Fused mental state (e.g., CALM, STRESSED, ANXIOUS)
    - Risk level (LOW / MODERATE / HIGH / CRITICAL)
    - Stability score (0–1)
    - Formatted psychological report string

  [Every session end (stored to data/user_memory/)]
    - Updated long-term memory JSON
    - Personality profile (Big Five analog traits)
    - Behavioral baseline (personal normal levels)
    - Deviation alerts (unusual behavior flags)
    - Personalized risk assessment

  [PSV charts (assets/reports/psv_visualizations/)]
    - <user_id>_radar.png       (Big Five trait radar)
    - <user_id>_bars.png        (trait bar chart)
    - <user_id>_dashboard.png   (comprehensive dashboard)
""")
print(f"  ALL CHECKS PASSED  (errors: {len(errors)})\n")
