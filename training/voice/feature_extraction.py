"""
Feature Extraction for Voice Emotion Recognition
=================================================
Phase 2: THE SOUL OF VOICE EMOTION DETECTION

Feature Families:
  1. Emotional Features: pitch, pitch variance, energy, speaking rate
  2. Stress Indicators: jitter, shimmer, spectral flatness, pauses

Philosophy:
  Turning sound into emotion fingerprints.
  These are the features humans subconsciously react to.

Author: Psychologist AI Team
Phase: 2 (Voice Emotion Recognition)
"""

import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')


# ============================================
# EMOTIONAL FEATURES
# ============================================

def extract_pitch_features(audio, sr, frame_length=2048, hop_length=512):
    """
    Extract pitch (F0) and pitch-related features.
    
    Pitch carries emotional information:
      - High pitch → excitement, fear
      - Low pitch → sadness, calmness
      - Pitch variation → emotional intensity
    
    Args:
        audio: Audio signal
        sr: Sample rate
        frame_length: Frame length for analysis
        hop_length: Hop length between frames
    
    Returns:
        pitch_features: Dictionary of pitch features
    """
    # Extract pitch using YIN algorithm
    pitches, magnitudes = librosa.piptrack(
        y=audio,
        sr=sr,
        fmin=75,   # Min pitch (human voice)
        fmax=500,  # Max pitch (human voice)
        n_fft=frame_length,
        hop_length=hop_length
    )
    
    # Get pitch contour (select strongest pitch per frame)
    pitch_contour = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:  # Valid pitch
            pitch_contour.append(pitch)
    
    if len(pitch_contour) == 0:
        # No valid pitch found
        return {
            'pitch_mean': 0,
            'pitch_std': 0,
            'pitch_range': 0,
            'pitch_median': 0,
            'pitch_slope': 0
        }
    
    pitch_contour = np.array(pitch_contour)
    
    # Calculate pitch slope with error handling
    pitch_slope = 0
    if len(pitch_contour) > 1:
        try:
            # Remove any NaN or inf values
            valid_pitch = pitch_contour[np.isfinite(pitch_contour)]
            if len(valid_pitch) > 1:
                pitch_slope = np.polyfit(range(len(valid_pitch)), valid_pitch, 1)[0]
        except (np.linalg.LinAlgError, ValueError):
            # If polyfit fails, use simple slope
            pitch_slope = (pitch_contour[-1] - pitch_contour[0]) / len(pitch_contour)
    
    # Calculate pitch statistics
    features = {
        'pitch_mean': float(np.mean(pitch_contour)),
        'pitch_std': float(np.std(pitch_contour)),
        'pitch_range': float(np.max(pitch_contour) - np.min(pitch_contour)),
        'pitch_median': float(np.median(pitch_contour)),
        'pitch_slope': float(pitch_slope)
    }
    
    return features


def extract_energy_features(audio, sr, frame_length=2048, hop_length=512):
    """
    Extract energy and intensity features.
    
    Energy indicates:
      - High energy → anger, happiness
      - Low energy → sadness, calmness
      - Energy variation → emotional dynamics
    
    Args:
        audio: Audio signal
        sr: Sample rate
        frame_length: Frame length
        hop_length: Hop length
    
    Returns:
        energy_features: Dictionary of energy features
    """
    # RMS energy
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    # Zero-crossing rate (voice activity indicator)
    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    features = {
        'rms_mean': np.mean(rms),
        'rms_std': np.std(rms),
        'rms_max': np.max(rms),
        'zcr_mean': np.mean(zcr),
        'zcr_std': np.std(zcr)
    }
    
    return features


def extract_spectral_features(audio, sr, n_mfcc=13):
    """
    Extract spectral features (MFCCs, spectral centroid, etc.).
    
    Spectral features capture voice timbre and quality.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCCs to extract
    
    Returns:
        spectral_features: Dictionary of spectral features
    """
    # MFCCs (standard voice features)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    # Spectral centroid (brightness)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    
    # Spectral rolloff (frequency distribution)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    
    # Spectral flatness (noisiness)
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
    
    features = {
        'mfcc_mean': mfcc_mean,
        'mfcc_std': mfcc_std,
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_centroid_std': np.std(spectral_centroid),
        'spectral_rolloff_mean': np.mean(spectral_rolloff),
        'spectral_rolloff_std': np.std(spectral_rolloff),
        'spectral_flatness_mean': np.mean(spectral_flatness),
        'spectral_flatness_std': np.std(spectral_flatness)
    }
    
    return features


def extract_temporal_features(audio, sr):
    """
    Extract temporal features (speaking rate, pauses).
    
    Temporal patterns indicate:
      - Fast speech → excitement, anxiety
      - Slow speech → sadness, calmness
      - Many pauses → hesitation, stress
    
    Args:
        audio: Audio signal
        sr: Sample rate
    
    Returns:
        temporal_features: Dictionary of temporal features
    """
    # Onset detection (speech events)
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
    
    # Speaking rate (onsets per second)
    speaking_rate = len(onset_frames) / (len(audio) / sr)
    
    # Detect pauses (silence regions)
    rms = librosa.feature.rms(y=audio)[0]
    silence_threshold = np.mean(rms) * 0.1
    silent_frames = rms < silence_threshold
    
    # Pause statistics
    pause_ratio = np.sum(silent_frames) / len(silent_frames)
    
    features = {
        'speaking_rate': speaking_rate,
        'num_onsets': len(onset_frames),
        'pause_ratio': pause_ratio
    }
    
    return features


# ============================================
# STRESS INDICATORS
# ============================================

def extract_jitter(pitch_contour):
    """
    Calculate jitter (pitch instability).
    
    High jitter indicates:
      - Vocal strain
      - Stress
      - Anxiety
    
    Args:
        pitch_contour: Array of pitch values
    
    Returns:
        jitter: Pitch period perturbation (%)
    """
    if len(pitch_contour) < 2:
        return 0.0
    
    # Remove zeros
    valid_pitches = pitch_contour[pitch_contour > 0]
    
    if len(valid_pitches) < 2:
        return 0.0
    
    # Calculate period differences
    periods = 1.0 / valid_pitches
    period_diffs = np.abs(np.diff(periods))
    
    # Jitter as percentage
    jitter = (np.mean(period_diffs) / np.mean(periods)) * 100
    
    return jitter


def extract_shimmer(audio, sr, frame_length=2048, hop_length=512):
    """
    Calculate shimmer (amplitude instability).
    
    High shimmer indicates:
      - Voice quality issues
      - Emotional arousal
      - Stress
    
    Args:
        audio: Audio signal
        sr: Sample rate
        frame_length: Frame length
        hop_length: Hop length
    
    Returns:
        shimmer: Amplitude perturbation (%)
    """
    # Calculate frame-wise RMS energy
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    if len(rms) < 2:
        return 0.0
    
    # Calculate amplitude differences
    rms_diffs = np.abs(np.diff(rms))
    
    # Shimmer as percentage
    shimmer = (np.mean(rms_diffs) / np.mean(rms)) * 100
    
    return shimmer


def extract_stress_features(audio, sr):
    """
    Extract all stress-related features.
    
    These are SEPARATE from emotion features.
    Stress is modeled independently.
    
    Args:
        audio: Audio signal
        sr: Sample rate
    
    Returns:
        stress_features: Dictionary of stress indicators
    """
    # Get pitch contour for jitter
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=75, fmax=500)
    pitch_contour = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_contour.append(pitch)
    pitch_contour = np.array(pitch_contour)
    
    # Calculate stress indicators
    jitter = extract_jitter(pitch_contour)
    shimmer = extract_shimmer(audio, sr)
    
    # Spectral flatness (voice quality)
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
    flatness_mean = np.mean(spectral_flatness)
    
    # Pitch variance (vocal stability)
    pitch_variance = np.std(pitch_contour) if len(pitch_contour) > 0 else 0
    
    features = {
        'jitter': jitter,
        'shimmer': shimmer,
        'spectral_flatness': flatness_mean,
        'pitch_variance': pitch_variance
    }
    
    return features


# ============================================
# COMPLETE FEATURE EXTRACTION
# ============================================

def extract_all_features(audio, sr, include_mfcc=True):
    """
    Extract all features for voice emotion recognition.
    
    Returns a comprehensive feature vector combining:
      - Emotional features (pitch, energy, temporal)
      - Spectral features (MFCCs, centroids)
      - Stress indicators (jitter, shimmer)
    
    Args:
        audio: Preprocessed audio signal
        sr: Sample rate
        include_mfcc: Whether to include MFCCs
    
    Returns:
        features: Dictionary of all features
        feature_vector: Flattened numpy array for model input
    """
    features = {}
    
    # Emotional features
    features.update(extract_pitch_features(audio, sr))
    features.update(extract_energy_features(audio, sr))
    features.update(extract_temporal_features(audio, sr))
    
    # Spectral features
    spectral = extract_spectral_features(audio, sr)
    
    # Handle MFCC arrays
    if include_mfcc:
        features['mfcc_mean'] = spectral['mfcc_mean']  # Array
        features['mfcc_std'] = spectral['mfcc_std']    # Array
    
    # Add other spectral features (scalars)
    for key in ['spectral_centroid_mean', 'spectral_centroid_std',
                'spectral_rolloff_mean', 'spectral_rolloff_std',
                'spectral_flatness_mean', 'spectral_flatness_std']:
        features[key] = spectral[key]
    
    # Stress indicators
    features.update(extract_stress_features(audio, sr))
    
    # Create feature vector (flatten all scalars + MFCC)
    feature_vector = []
    
    # Add scalar features
    scalar_keys = [
        'pitch_mean', 'pitch_std', 'pitch_range', 'pitch_median', 'pitch_slope',
        'rms_mean', 'rms_std', 'rms_max', 'zcr_mean', 'zcr_std',
        'speaking_rate', 'num_onsets', 'pause_ratio',
        'spectral_centroid_mean', 'spectral_centroid_std',
        'spectral_rolloff_mean', 'spectral_rolloff_std',
        'spectral_flatness_mean', 'spectral_flatness_std',
        'jitter', 'shimmer', 'pitch_variance'
    ]
    
    for key in scalar_keys:
        value = features.get(key, 0.0)
        # Ensure value is finite (not NaN or inf)
        if not np.isfinite(value):
            value = 0.0
        feature_vector.append(float(value))
    
    # Add MFCC features
    if include_mfcc:
        mfcc_mean = features['mfcc_mean']
        mfcc_std = features['mfcc_std']
        
        # Ensure MFCC values are finite and flatten to 1D
        mfcc_mean = np.nan_to_num(mfcc_mean, nan=0.0, posinf=0.0, neginf=0.0)
        mfcc_std = np.nan_to_num(mfcc_std, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Flatten and convert to list to avoid shape issues
        mfcc_mean_flat = np.array(mfcc_mean).flatten().tolist()
        mfcc_std_flat = np.array(mfcc_std).flatten().tolist()
        
        feature_vector.extend(mfcc_mean_flat)
        feature_vector.extend(mfcc_std_flat)
    
    # Convert to numpy array with explicit dtype and shape check
    feature_vector = np.array(feature_vector, dtype=np.float32).flatten()
    
    return features, feature_vector


def extract_stress_score(audio, sr):
    """
    Calculate overall stress score from audio.
    
    Returns:
      - Binary: stressed / not stressed
      - Scalar: 0-1 stress level
      - Categorical: low / medium / high
    
    Args:
        audio: Preprocessed audio signal
        sr: Sample rate
    
    Returns:
        stress_dict: Dictionary with stress metrics
    """
    stress_features = extract_stress_features(audio, sr)
    
    # Normalize features to 0-1 range (approximate)
    jitter_norm = min(stress_features['jitter'] / 2.0, 1.0)  # Normal < 1%
    shimmer_norm = min(stress_features['shimmer'] / 10.0, 1.0)  # Normal < 5%
    flatness_norm = stress_features['spectral_flatness']  # Already 0-1
    pitch_var_norm = min(stress_features['pitch_variance'] / 50.0, 1.0)  # Normal < 25
    
    # Weighted stress score
    stress_score = (
        0.3 * jitter_norm +
        0.3 * shimmer_norm +
        0.2 * flatness_norm +
        0.2 * pitch_var_norm
    )
    
    # Categorical stress level
    if stress_score < 0.3:
        stress_level = 'low'
    elif stress_score < 0.6:
        stress_level = 'medium'
    else:
        stress_level = 'high'
    
    # Binary stress flag
    is_stressed = stress_score > 0.5
    
    return {
        'stress_score': stress_score,
        'stress_level': stress_level,
        'is_stressed': is_stressed,
        'jitter': stress_features['jitter'],
        'shimmer': stress_features['shimmer'],
        'spectral_flatness': stress_features['spectral_flatness'],
        'pitch_variance': stress_features['pitch_variance']
    }


# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_feature_names(include_mfcc=True):
    """Get list of feature names in order."""
    names = [
        'pitch_mean', 'pitch_std', 'pitch_range', 'pitch_median', 'pitch_slope',
        'rms_mean', 'rms_std', 'rms_max', 'zcr_mean', 'zcr_std',
        'speaking_rate', 'num_onsets', 'pause_ratio',
        'spectral_centroid_mean', 'spectral_centroid_std',
        'spectral_rolloff_mean', 'spectral_rolloff_std',
        'spectral_flatness_mean', 'spectral_flatness_std',
        'jitter', 'shimmer', 'pitch_variance'
    ]
    
    if include_mfcc:
        names.extend([f'mfcc_{i}_mean' for i in range(13)])
        names.extend([f'mfcc_{i}_std' for i in range(13)])
    
    return names


def get_feature_dimension(include_mfcc=True):
    """Get total feature vector dimension."""
    base_features = 22  # Scalar features
    mfcc_features = 26 if include_mfcc else 0  # 13 mean + 13 std
    return base_features + mfcc_features


if __name__ == "__main__":
    # Test feature extraction
    print("=" * 60)
    print("FEATURE EXTRACTION TEST")
    print("=" * 60)
    
    # Generate test audio (1 second of mixed tones)
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Simulate speech-like signal
    test_audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +  # Base pitch
        0.2 * np.sin(2 * np.pi * 400 * t) +  # Harmonic
        0.05 * np.random.randn(len(t))       # Noise
    )
    
    print(f"\n[Test] Generated test audio: {duration}s at {sr}Hz")
    
    # Extract features
    print("\n[Extracting] All features...")
    features, feature_vector = extract_all_features(test_audio, sr)
    
    print(f"\n[Result] Feature vector dimension: {len(feature_vector)}")
    print(f"[Result] Feature names: {get_feature_dimension()} features")
    
    # Show some key features
    print("\n[Emotional Features]")
    print(f"  Pitch mean: {features['pitch_mean']:.2f} Hz")
    print(f"  Pitch std: {features['pitch_std']:.2f} Hz")
    print(f"  RMS energy: {features['rms_mean']:.4f}")
    print(f"  Speaking rate: {features['speaking_rate']:.2f} events/s")
    
    # Extract stress score
    print("\n[Stress Features]")
    stress = extract_stress_score(test_audio, sr)
    print(f"  Stress score: {stress['stress_score']:.3f}")
    print(f"  Stress level: {stress['stress_level']}")
    print(f"  Jitter: {stress['jitter']:.3f}%")
    print(f"  Shimmer: {stress['shimmer']:.3f}%")
    
    print("\n" + "=" * 60)
    print("✓ Feature extraction ready!")
    print(f"✓ Feature dimension: {get_feature_dimension()}")
    print("=" * 60)
