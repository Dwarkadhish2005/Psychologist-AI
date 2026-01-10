"""
Audio Preprocessing for Voice Emotion Recognition
==================================================
Phase 2: Critical preprocessing pipeline for audio input

Goals:
  - Remove silence
  - Normalize volume
  - Reduce noise
  - Standardize sampling rate

Philosophy:
  Emotion lives in variation, not loudness.
  Stress lives in instability.
  If preprocessing is bad → model learns junk.

Author: Psychologist AI Team
Phase: 2 (Voice Emotion Recognition)
"""

import numpy as np
import librosa
import noisereduce as nr
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


# ============================================
# CORE AUDIO PREPROCESSING
# ============================================

def load_audio(file_path, target_sr=16000):
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sampling rate (default 16kHz)
    
    Returns:
        audio: Audio signal (mono)
        sr: Sample rate
    """
    # Load audio (librosa automatically converts to mono)
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr


def remove_silence(audio, sr, threshold_db=-40, frame_length=2048, hop_length=512):
    """
    Remove silence from audio signal.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        threshold_db: Silence threshold in dB
        frame_length: Frame length for analysis
        hop_length: Hop length between frames
    
    Returns:
        trimmed_audio: Audio with silence removed
    """
    # Trim silence from beginning and end
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=abs(threshold_db),
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    return trimmed


def normalize_audio(audio, target_rms=0.1):
    """
    Normalize audio volume to target RMS level.
    
    Why: Emotion lives in variation, not loudness.
    
    Args:
        audio: Audio signal
        target_rms: Target RMS level (default 0.1)
    
    Returns:
        normalized_audio: Volume-normalized audio
    """
    # Calculate current RMS
    current_rms = np.sqrt(np.mean(audio**2))
    
    # Avoid division by zero
    if current_rms < 1e-6:
        return audio
    
    # Normalize to target RMS
    normalized = audio * (target_rms / current_rms)
    
    # Clip to [-1, 1] to avoid distortion
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized


def reduce_noise(audio, sr, stationary=True):
    """
    Reduce background noise from audio.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        stationary: Whether noise is stationary (default True)
    
    Returns:
        denoised_audio: Noise-reduced audio
    """
    try:
        # Use noisereduce library for spectral gating
        denoised = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=stationary,
            prop_decrease=0.8  # Reduce noise by 80%
        )
        return denoised
    except Exception as e:
        print(f"[Warning] Noise reduction failed: {e}")
        return audio  # Return original if fails


def apply_preemphasis(audio, coef=0.97):
    """
    Apply pre-emphasis filter to enhance high frequencies.
    
    Helps with feature extraction (especially pitch).
    
    Args:
        audio: Audio signal
        coef: Pre-emphasis coefficient (default 0.97)
    
    Returns:
        emphasized_audio: Pre-emphasized audio
    """
    emphasized = np.append(audio[0], audio[1:] - coef * audio[:-1])
    return emphasized


def segment_audio(audio, sr, window_duration=3.0, hop_duration=1.5):
    """
    Segment audio into overlapping windows.
    
    Why: Voice emotion is temporal, not static.
         Overlapping chunks capture emotion changes.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        window_duration: Window duration in seconds
        hop_duration: Hop duration in seconds
    
    Returns:
        segments: List of audio segments
        timestamps: Start time of each segment
    """
    window_samples = int(window_duration * sr)
    hop_samples = int(hop_duration * sr)
    
    segments = []
    timestamps = []
    
    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        end = start + window_samples
        segment = audio[start:end]
        segments.append(segment)
        timestamps.append(start / sr)
    
    # Add final segment if audio is longer
    if len(audio) > window_samples and (len(audio) - window_samples) % hop_samples != 0:
        segments.append(audio[-window_samples:])
        timestamps.append((len(audio) - window_samples) / sr)
    
    return segments, timestamps


def check_audio_quality(audio, sr, min_duration=0.5):
    """
    Check if audio is suitable for emotion recognition.
    
    Returns quality flags and metrics.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        min_duration: Minimum duration in seconds
    
    Returns:
        quality_dict: Dictionary with quality metrics
    """
    quality = {
        'is_valid': True,
        'duration': len(audio) / sr,
        'rms_energy': np.sqrt(np.mean(audio**2)),
        'peak_amplitude': np.max(np.abs(audio)),
        'warnings': []
    }
    
    # Check duration
    if quality['duration'] < min_duration:
        quality['is_valid'] = False
        quality['warnings'].append(f"Audio too short ({quality['duration']:.2f}s < {min_duration}s)")
    
    # Check if audio is too quiet
    if quality['rms_energy'] < 0.001:
        quality['warnings'].append("Audio too quiet (low RMS)")
    
    # Check if audio is clipping
    if quality['peak_amplitude'] > 0.99:
        quality['warnings'].append("Audio clipping detected")
    
    # Check if audio is mostly silence
    non_silent = np.sum(np.abs(audio) > 0.01)
    if non_silent / len(audio) < 0.1:
        quality['warnings'].append("Audio mostly silent")
        quality['is_valid'] = False
    
    return quality


# ============================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================

def preprocess_audio(audio_input, sr=None, target_sr=16000, 
                     remove_silence_flag=True, normalize_flag=True, 
                     denoise_flag=True, preemphasis_flag=True):
    """
    Complete audio preprocessing pipeline.
    
    This is the STANDARD pipeline - use everywhere!
    
    Args:
        audio_input: Audio file path or numpy array
        sr: Sample rate (required if audio_input is array)
        target_sr: Target sample rate
        remove_silence_flag: Whether to remove silence
        normalize_flag: Whether to normalize volume
        denoise_flag: Whether to reduce noise
        preemphasis_flag: Whether to apply pre-emphasis
    
    Returns:
        processed_audio: Preprocessed audio signal
        sr: Sample rate
        quality: Quality metrics dictionary
    """
    # Load audio if file path
    if isinstance(audio_input, str):
        audio, sr = load_audio(audio_input, target_sr=target_sr)
    else:
        audio = audio_input
        if sr is None:
            raise ValueError("Sample rate must be provided for array input")
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
    
    # Check initial quality
    quality = check_audio_quality(audio, sr)
    
    if not quality['is_valid']:
        print(f"[Warning] Audio quality issues: {quality['warnings']}")
        return audio, sr, quality
    
    # Apply preprocessing steps
    if remove_silence_flag:
        audio = remove_silence(audio, sr)
    
    if denoise_flag:
        audio = reduce_noise(audio, sr)
    
    if normalize_flag:
        audio = normalize_audio(audio)
    
    if preemphasis_flag:
        audio = apply_preemphasis(audio)
    
    # Final quality check
    quality = check_audio_quality(audio, sr)
    
    return audio, sr, quality


def preprocess_realtime_chunk(audio_chunk, sr=16000):
    """
    Fast preprocessing for real-time audio chunks.
    
    Optimized for low latency (microphone input).
    
    Args:
        audio_chunk: Audio chunk from microphone
        sr: Sample rate
    
    Returns:
        processed_chunk: Preprocessed audio
    """
    # Quick quality check
    if len(audio_chunk) == 0 or np.max(np.abs(audio_chunk)) < 0.001:
        return None  # Too quiet or empty
    
    # Normalize only (fastest)
    processed = normalize_audio(audio_chunk)
    
    # Optional: light noise reduction
    # (disable if too slow for real-time)
    # processed = reduce_noise(processed, sr, stationary=True)
    
    return processed


# ============================================
# UTILITY FUNCTIONS
# ============================================

def save_audio(audio, sr, output_path):
    """Save audio to file."""
    import soundfile as sf
    sf.write(output_path, audio, sr)
    print(f"[Save] Audio saved to: {output_path}")


def get_audio_info(file_path):
    """Get audio file information."""
    import soundfile as sf
    info = sf.info(file_path)
    return {
        'duration': info.duration,
        'sample_rate': info.samplerate,
        'channels': info.channels,
        'format': info.format
    }


if __name__ == "__main__":
    # Test preprocessing pipeline
    print("=" * 60)
    print("AUDIO PREPROCESSING TEST")
    print("=" * 60)
    
    # Generate test audio (1 second of 440Hz tone)
    duration = 1.0
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note
    
    # Add some noise
    test_audio += 0.05 * np.random.randn(len(test_audio))
    
    print(f"\n[Test] Generated test audio: {duration}s at {sr}Hz")
    print(f"[Test] Original RMS: {np.sqrt(np.mean(test_audio**2)):.4f}")
    
    # Preprocess
    processed, sr_out, quality = preprocess_audio(
        test_audio, 
        sr=sr,
        target_sr=16000
    )
    
    print(f"\n[Result] Processed audio: {len(processed)/sr_out:.2f}s")
    print(f"[Result] Processed RMS: {np.sqrt(np.mean(processed**2)):.4f}")
    print(f"[Result] Quality: {quality}")
    
    print("\n" + "=" * 60)
    print("✓ Preprocessing pipeline ready!")
    print("=" * 60)
