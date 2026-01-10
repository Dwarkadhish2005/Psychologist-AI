"""
Dataset Downloader for Voice Emotion Recognition
=================================================
Phase 2: Download and prepare RAVDESS and TESS datasets

Recommended Combination:
  🥇 RAVDESS → Primary dataset
  🥈 TESS → Emotion reinforcement
  🥉 CREMA-D → Optional (later)

Author: Psychologist AI Team
Phase: 2 (Voice Emotion Recognition)
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import json
from dataset_utils import (
    map_ravdess_label, map_tess_label, derive_stress_from_emotion,
    create_speaker_independent_split, balance_dataset_by_emotion,
    print_dataset_info, TARGET_EMOTIONS
)


# ============================================
# DATASET PATHS AND URLs
# ============================================

# Update these paths for your system
DATA_ROOT = Path(r"C:\Dwarka\Machiene Learning\Psycologist AI\data\voice_emotion")

# RAVDESS dataset info
RAVDESS_INFO = {
    'name': 'RAVDESS',
    'url': 'https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip',
    'description': 'Ryerson Audio-Visual Database of Emotional Speech and Song',
    'num_actors': 24,
    'emotions': ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
    'intensities': ['normal', 'strong']
}

# TESS dataset info
TESS_INFO = {
    'name': 'TESS',
    'url': 'https://tspace.library.utoronto.ca/bitstream/1807/24487/1/TESS_Toronto_emotional_speech_set_data.zip',
    'description': 'Toronto Emotional Speech Set',
    'num_speakers': 2,
    'emotions': ['angry', 'disgust', 'fear', 'happy', 'pleasant_surprise', 'sad', 'neutral']
}


# ============================================
# RAVDESS PARSER
# ============================================

def parse_ravdess_filename(filename):
    """
    Parse RAVDESS filename to extract metadata.
    
    RAVDESS filename format:
    Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    
    Example: 03-01-06-01-02-01-12.wav
      - 03: Speech
      - 01: Full audio
      - 06: Fearful (emotion)
      - 01: Normal intensity
      - 02: Statement "dogs"
      - 01: 1st repetition
      - 12: Actor 12
    
    Args:
        filename: RAVDESS filename
    
    Returns:
        metadata: Dictionary with parsed information
    """
    parts = filename.replace('.wav', '').split('-')
    
    if len(parts) != 7:
        return None
    
    modality, vocal_channel, emotion_code, intensity_code, statement, repetition, actor = parts
    
    # Emotion mapping
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    # Intensity mapping
    intensity_map = {
        '01': 'normal',
        '02': 'strong'
    }
    
    # Get original emotion
    original_emotion = emotion_map.get(emotion_code, 'unknown')
    
    # Map to target emotion
    target_emotion, note = map_ravdess_label(original_emotion)
    
    if target_emotion is None:
        return None
    
    # Derive stress
    stress_level, stress_score = derive_stress_from_emotion(target_emotion, source_label=original_emotion)
    
    # Actor gender (odd=male, even=female)
    actor_num = int(actor)
    gender = 'male' if actor_num % 2 == 1 else 'female'
    
    metadata = {
        'file_name': filename,
        'speaker_id': f'ravdess_actor_{actor}',
        'gender': gender,
        'original_emotion': original_emotion,
        'emotion': target_emotion,
        'intensity': intensity_map.get(intensity_code, 'unknown'),
        'stress_level': stress_level,
        'stress_score': stress_score,
        'statement': statement,
        'repetition': repetition,
        'mapping_note': note
    }
    
    return metadata


def load_ravdess_dataset(ravdess_path):
    """
    Load and parse RAVDESS dataset.
    
    Args:
        ravdess_path: Path to RAVDESS dataset directory
    
    Returns:
        dataset: List of sample dictionaries
    """
    dataset = []
    
    ravdess_path = Path(ravdess_path)
    
    # RAVDESS has actor folders (Actor_01, Actor_02, ...)
    actor_folders = sorted(ravdess_path.glob('Actor_*'))
    
    print(f"\n[RAVDESS] Found {len(actor_folders)} actor folders")
    
    for actor_folder in actor_folders:
        wav_files = list(actor_folder.glob('*.wav'))
        
        for wav_file in wav_files:
            metadata = parse_ravdess_filename(wav_file.name)
            
            if metadata is not None:
                metadata['file_path'] = str(wav_file)
                metadata['dataset'] = 'RAVDESS'
                dataset.append(metadata)
    
    print(f"[RAVDESS] Loaded {len(dataset)} samples")
    
    return dataset


# ============================================
# TESS PARSER
# ============================================

def parse_tess_filename(filename):
    """
    Parse TESS filename to extract metadata.
    
    TESS filename format:
    {speaker}_{word}_{emotion}.wav
    
    Example: OAF_back_angry.wav
      - OAF: Older Adult Female speaker
      - back: Word spoken
      - angry: Emotion
    
    Args:
        filename: TESS filename
    
    Returns:
        metadata: Dictionary with parsed information
    """
    parts = filename.replace('.wav', '').split('_')
    
    if len(parts) < 3:
        return None
    
    speaker = parts[0]
    word = parts[1] if len(parts) >= 2 else 'unknown'
    original_emotion = parts[-1]  # Last part is emotion
    
    # Map to target emotion
    target_emotion, note = map_tess_label(original_emotion)
    
    if target_emotion is None:
        return None
    
    # Derive stress
    stress_level, stress_score = derive_stress_from_emotion(target_emotion)
    
    # Speaker info
    speaker_map = {
        'OAF': ('older_adult_female', 'female'),
        'YAF': ('younger_adult_female', 'female')
    }
    
    speaker_desc, gender = speaker_map.get(speaker, ('unknown', 'unknown'))
    
    metadata = {
        'file_name': filename,
        'speaker_id': f'tess_{speaker}',
        'gender': gender,
        'speaker_desc': speaker_desc,
        'word': word,
        'original_emotion': original_emotion,
        'emotion': target_emotion,
        'stress_level': stress_level,
        'stress_score': stress_score,
        'mapping_note': note
    }
    
    return metadata


def load_tess_dataset(tess_path):
    """
    Load and parse TESS dataset.
    
    Args:
        tess_path: Path to TESS dataset directory
    
    Returns:
        dataset: List of sample dictionaries
    """
    dataset = []
    
    tess_path = Path(tess_path)
    
    # TESS might have emotion folders or flat structure
    wav_files = list(tess_path.rglob('*.wav'))
    
    print(f"\n[TESS] Found {len(wav_files)} audio files")
    
    for wav_file in wav_files:
        metadata = parse_tess_filename(wav_file.name)
        
        if metadata is not None:
            metadata['file_path'] = str(wav_file)
            metadata['dataset'] = 'TESS'
            dataset.append(metadata)
    
    print(f"[TESS] Loaded {len(dataset)} samples")
    
    return dataset


# ============================================
# DATASET DOWNLOADER
# ============================================

def download_file(url, output_path, chunk_size=8192):
    """
    Download file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save file
        chunk_size: Download chunk size
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Download] Downloading from {url}")
    print(f"[Download] Saving to {output_path}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=output_path.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))
    
    print(f"[Download] Complete!")


def extract_zip(zip_path, extract_to):
    """
    Extract zip file.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    print(f"\n[Extract] Extracting {zip_path}")
    print(f"[Extract] To {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"[Extract] Complete!")


def download_ravdess():
    """
    Download RAVDESS dataset.
    
    Note: RAVDESS is ~1GB. This might take a while.
    """
    ravdess_dir = DATA_ROOT / 'RAVDESS'
    ravdess_zip = DATA_ROOT / 'ravdess.zip'
    
    if ravdess_dir.exists():
        print(f"[RAVDESS] Already downloaded at {ravdess_dir}")
        return ravdess_dir
    
    # Download
    if not ravdess_zip.exists():
        download_file(RAVDESS_INFO['url'], ravdess_zip)
    
    # Extract
    extract_zip(ravdess_zip, ravdess_dir)
    
    return ravdess_dir


def download_tess():
    """
    Download TESS dataset.
    
    Note: TESS is smaller (~500MB).
    """
    tess_dir = DATA_ROOT / 'TESS'
    tess_zip = DATA_ROOT / 'tess.zip'
    
    if tess_dir.exists():
        print(f"[TESS] Already downloaded at {tess_dir}")
        return tess_dir
    
    # Download
    if not tess_zip.exists():
        download_file(TESS_INFO['url'], tess_zip)
    
    # Extract
    extract_zip(tess_zip, tess_dir)
    
    return tess_dir


# ============================================
# DATASET PREPARATION
# ============================================

def prepare_combined_dataset(use_ravdess=True, use_tess=True, balance_strategy='equal'):
    """
    Prepare combined dataset from RAVDESS and TESS.
    
    Strategy:
      1. Load both datasets
      2. Combine them
      3. Create speaker-independent splits
      4. Balance emotions
      5. Save metadata
    
    Args:
        use_ravdess: Whether to include RAVDESS
        use_tess: Whether to include TESS
        balance_strategy: 'equal', 'upsample', or 'weighted'
    
    Returns:
        train_data: Training set
        val_data: Validation set
        test_data: Test set
    """
    all_data = []
    
    # Load RAVDESS
    if use_ravdess:
        ravdess_path = DATA_ROOT / 'RAVDESS'
        if ravdess_path.exists():
            ravdess_data = load_ravdess_dataset(ravdess_path)
            all_data.extend(ravdess_data)
        else:
            print(f"[Warning] RAVDESS not found at {ravdess_path}")
            print(f"[Warning] Run download_ravdess() first or set use_ravdess=False")
    
    # Load TESS
    if use_tess:
        tess_path = DATA_ROOT / 'TESS'
        if tess_path.exists():
            tess_data = load_tess_dataset(tess_path)
            all_data.extend(tess_data)
        else:
            print(f"[Warning] TESS not found at {tess_path}")
            print(f"[Warning] Run download_tess() first or set use_tess=False")
    
    if len(all_data) == 0:
        print("[Error] No data loaded!")
        return None, None, None
    
    # Print combined statistics
    print_dataset_info(all_data, "Combined Dataset")
    
    # Speaker-independent split
    print("\n[Split] Creating speaker-independent splits...")
    train_data, val_data, test_data = create_speaker_independent_split(
        all_data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Balance training data
    print(f"\n[Balance] Balancing training data ({balance_strategy})...")
    train_data, class_weights = balance_dataset_by_emotion(train_data, strategy=balance_strategy)
    
    # Print split statistics
    print_dataset_info(train_data, "Training Set")
    print_dataset_info(val_data, "Validation Set")
    print_dataset_info(test_data, "Test Set")
    
    # Save metadata
    metadata = {
        'datasets': [],
        'num_train': len(train_data),
        'num_val': len(val_data),
        'num_test': len(test_data),
        'balance_strategy': balance_strategy,
        'class_weights': class_weights if class_weights else {}
    }
    
    if use_ravdess:
        metadata['datasets'].append('RAVDESS')
    if use_tess:
        metadata['datasets'].append('TESS')
    
    metadata_path = DATA_ROOT / 'dataset_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[Save] Metadata saved to {metadata_path}")
    
    # Save splits
    splits = {
        'train': [item['file_path'] for item in train_data],
        'val': [item['file_path'] for item in val_data],
        'test': [item['file_path'] for item in test_data]
    }
    
    splits_path = DATA_ROOT / 'dataset_splits.json'
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"[Save] Splits saved to {splits_path}")
    
    return train_data, val_data, test_data


# ============================================
# MANUAL DATASET SETUP INSTRUCTIONS
# ============================================

def print_manual_setup_instructions():
    """
    Print instructions for manual dataset download.
    
    (Auto-download might fail due to authentication/bandwidth)
    """
    print("\n" + "=" * 60)
    print("MANUAL DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    
    print("\n🥇 RAVDESS (Primary Dataset)")
    print("-" * 60)
    print("1. Go to: https://zenodo.org/record/1188976")
    print("2. Download: 'Audio_Speech_Actors_01-24.zip' (~1GB)")
    print(f"3. Extract to: {DATA_ROOT / 'RAVDESS'}")
    print("4. Folder structure should be:")
    print("   RAVDESS/")
    print("     Actor_01/")
    print("       03-01-01-01-01-01-01.wav")
    print("       ...")
    print("     Actor_02/")
    print("       ...")
    
    print("\n🥈 TESS (Secondary Dataset)")
    print("-" * 60)
    print("1. Go to: https://tspace.library.utoronto.ca/handle/1807/24487")
    print("2. Download the dataset (~500MB)")
    print(f"3. Extract to: {DATA_ROOT / 'TESS'}")
    print("4. Folder structure should be:")
    print("   TESS/")
    print("     OAF_angry/")
    print("       OAF_back_angry.wav")
    print("       ...")
    print("     YAF_happy/")
    print("       ...")
    
    print("\n" + "=" * 60)
    print("After downloading, run:")
    print("  train_data, val_data, test_data = prepare_combined_dataset()")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("=" * 60)
    print("VOICE EMOTION DATASET DOWNLOADER")
    print("=" * 60)
    
    # Print manual setup instructions
    print_manual_setup_instructions()
    
    # Check if datasets exist
    ravdess_exists = (DATA_ROOT / 'RAVDESS').exists()
    tess_exists = (DATA_ROOT / 'TESS').exists()
    
    print(f"\n[Status] RAVDESS: {'✓ Found' if ravdess_exists else '✗ Not found'}")
    print(f"[Status] TESS: {'✓ Found' if tess_exists else '✗ Not found'}")
    
    if ravdess_exists or tess_exists:
        print("\n[Info] Datasets found! You can now run:")
        print("  prepare_combined_dataset(use_ravdess=True, use_tess=True)")
    else:
        print("\n[Info] Please download datasets manually using instructions above.")
