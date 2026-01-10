"""
🎯 PHASE 4.1 & 4.2 DEMO
======================

Demonstrates new features:
- User management and selection
- Session limits (max 10/day)
- Data export (CSV/JSON)
- Archive functionality

Usage:
    python inference/demo_phase4_enhancements.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.phase4_user_manager import UserSelector, UserManager
from inference.phase4_cognitive_layer import LongTermMemory, SessionMemory, SessionMetrics
from inference.phase3_multimodal_fusion import PsychologicalState, MentalState, RiskLevel
import time
import random


def demo_user_management():
    """Demo: User registration and selection"""
    print("\n" + "="*70)
    print("DEMO 1: USER MANAGEMENT")
    print("="*70)
    
    manager = UserManager(storage_dir="data/demo_memory")
    
    # Register some test users
    print("\n📝 Registering test users...")
    alice_id = manager.register_user("Alice")
    bob_id = manager.register_user("Bob")
    charlie_id = manager.register_user("Charlie")
    
    # List users
    print("\n📋 Registered users:")
    users = manager.list_users()
    for user in users:
        print(f"  - {user.name} (ID: {user.user_id})")
        print(f"    Created: {user.created_at}")
        print(f"    Sessions: {user.total_sessions}")
    
    # Simulate some activity
    print("\n🎬 Simulating user activity...")
    manager.increment_session_count(alice_id)
    manager.increment_session_count(alice_id)
    manager.increment_session_count(bob_id)
    
    print(f"  Alice: {manager.get_user(alice_id).total_sessions} sessions")
    print(f"  Bob: {manager.get_user(bob_id).total_sessions} sessions")
    
    return alice_id, bob_id, charlie_id


def demo_session_limits(user_id: str):
    """Demo: Session limits (max 10/day)"""
    print("\n" + "="*70)
    print("DEMO 2: SESSION LIMITS (Max 10/day)")
    print("="*70)
    
    ltm = LongTermMemory(
        user_id=user_id,
        storage_dir="data/demo_memory",
        max_sessions_per_day=5  # Lower for demo
    )
    
    print(f"\n📊 Adding sessions for user: {user_id}")
    print(f"   Session limit: {ltm.max_sessions_per_day}/day")
    
    # Create dummy session metrics
    for i in range(8):
        print(f"\n   Session {i+1}...")
        
        metrics = SessionMetrics(
            session_start=time.time(),
            session_duration=600,  # 10 minutes
            total_frames=300,
            dominant_mental_states=[(MentalState.CALM, 0.7), (MentalState.STRESSED, 0.3)],
            mental_state_switches=5,
            avg_confidence=0.75,
            avg_stability=0.80,
            confidence_variance=0.05,
            stability_variance=0.03,
            stress_duration_ratio=0.3,
            high_stress_duration_ratio=0.1,
            avg_stress_intensity=0.4,
            masking_frequency=2.0,
            total_masking_events=20,
            masking_duration_ratio=0.15,
            avg_risk_level=0.5,
            high_risk_duration_ratio=0.1,
            risk_escalations=1,
            positive_emotion_ratio=0.6,
            negative_emotion_ratio=0.2,
            neutral_emotion_ratio=0.2
        )
        
        ltm.add_session(metrics)
        
        if i >= ltm.max_sessions_per_day - 1:
            print(f"   ⚠️ Session limit reached!")
    
    # Show daily profile
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    if today in ltm.daily_profiles:
        profile = ltm.daily_profiles[today]
        print(f"\n✓ Daily profile for {today}:")
        print(f"   Total sessions: {profile.total_sessions}")
        print(f"   Duration: {profile.total_duration_minutes:.1f} minutes")


def demo_data_export(user_id: str):
    """Demo: Export to CSV and JSON"""
    print("\n" + "="*70)
    print("DEMO 3: DATA EXPORT (CSV & JSON)")
    print("="*70)
    
    ltm = LongTermMemory(
        user_id=user_id,
        storage_dir="data/demo_memory"
    )
    
    # Export to CSV
    print("\n📊 Exporting to CSV...")
    csv_path = ltm.export_to_csv()
    print(f"   ✓ Created: {csv_path}")
    
    # Show first few lines
    with open(csv_path, 'r') as f:
        lines = f.readlines()[:5]
        print(f"\n   Preview (first 4 rows):")
        for line in lines:
            print(f"   {line.strip()}")
    
    # Export to JSON
    print("\n📊 Exporting to JSON...")
    json_path = ltm.export_to_json()
    print(f"   ✓ Created: {json_path}")
    
    # Show file size
    from pathlib import Path
    json_size = Path(json_path).stat().st_size
    print(f"   File size: {json_size / 1024:.2f} KB")


def demo_archive_functionality(user_id: str):
    """Demo: Automatic archiving before deletion"""
    print("\n" + "="*70)
    print("DEMO 4: ARCHIVE FUNCTIONALITY")
    print("="*70)
    
    ltm = LongTermMemory(
        user_id=user_id,
        storage_dir="data/demo_memory",
        max_days_stored=7  # Short for demo
    )
    
    # Add sessions for multiple days (simulate 10 days)
    print("\n📅 Simulating 10 days of data...")
    from datetime import datetime, timedelta
    
    for day_offset in range(10):
        date = datetime.now() - timedelta(days=10 - day_offset)
        
        metrics = SessionMetrics(
            session_start=date.timestamp(),
            session_duration=600,
            total_frames=300,
            dominant_mental_states=[(MentalState.CALM, 0.7)],
            mental_state_switches=3,
            avg_confidence=0.75,
            avg_stability=0.80,
            confidence_variance=0.05,
            stability_variance=0.03,
            stress_duration_ratio=0.2,
            high_stress_duration_ratio=0.05,
            avg_stress_intensity=0.3,
            masking_frequency=1.5,
            total_masking_events=15,
            masking_duration_ratio=0.10,
            avg_risk_level=0.4,
            high_risk_duration_ratio=0.05,
            risk_escalations=0,
            positive_emotion_ratio=0.7,
            negative_emotion_ratio=0.1,
            neutral_emotion_ratio=0.2
        )
        
        ltm.add_session(metrics)
    
    print(f"   ✓ Created {len(ltm.daily_profiles)} days of data")
    
    # List archives
    archives = ltm.get_archive_list()
    if archives:
        print(f"\n📦 Archives created:")
        for archive in archives:
            print(f"   - {archive['filename']}")
            print(f"     Range: {archive['date_range']}")
            print(f"     Size: {archive['size_kb']:.2f} KB")
    else:
        print(f"\n   No archives yet (data within {ltm.max_days_stored} day window)")


def demo_multi_user_separation():
    """Demo: Multiple users with separate data"""
    print("\n" + "="*70)
    print("DEMO 5: MULTI-USER DATA SEPARATION")
    print("="*70)
    
    # Create two users with different patterns
    alice_ltm = LongTermMemory(user_id="alice_demo", storage_dir="data/demo_memory")
    bob_ltm = LongTermMemory(user_id="bob_demo", storage_dir="data/demo_memory")
    
    print("\n👤 Alice (calm user):")
    alice_metrics = SessionMetrics(
        session_start=time.time(),
        session_duration=1800,
        total_frames=900,
        dominant_mental_states=[(MentalState.CALM, 0.9), (MentalState.STRESSED, 0.1)],
        mental_state_switches=2,
        avg_confidence=0.85,
        avg_stability=0.90,
        confidence_variance=0.02,
        stability_variance=0.01,
        stress_duration_ratio=0.1,
        high_stress_duration_ratio=0.02,
        avg_stress_intensity=0.2,
        masking_frequency=0.5,
        total_masking_events=5,
        masking_duration_ratio=0.05,
        avg_risk_level=0.2,
        high_risk_duration_ratio=0.01,
        risk_escalations=0,
        positive_emotion_ratio=0.8,
        negative_emotion_ratio=0.1,
        neutral_emotion_ratio=0.1
    )
    alice_ltm.add_session(alice_metrics)
    
    print(f"   Stress: {alice_metrics.stress_duration_ratio*100:.1f}%")
    print(f"   Stability: {alice_metrics.avg_stability*100:.1f}%")
    
    print("\n👤 Bob (stressed user):")
    bob_metrics = SessionMetrics(
        session_start=time.time(),
        session_duration=1800,
        total_frames=900,
        dominant_mental_states=[(MentalState.STRESSED, 0.6), (MentalState.ANXIOUS, 0.3)],
        mental_state_switches=15,
        avg_confidence=0.60,
        avg_stability=0.50,
        confidence_variance=0.15,
        stability_variance=0.20,
        stress_duration_ratio=0.7,
        high_stress_duration_ratio=0.4,
        avg_stress_intensity=0.8,
        masking_frequency=5.0,
        total_masking_events=50,
        masking_duration_ratio=0.40,
        avg_risk_level=1.5,
        high_risk_duration_ratio=0.35,
        risk_escalations=3,
        positive_emotion_ratio=0.2,
        negative_emotion_ratio=0.6,
        neutral_emotion_ratio=0.2
    )
    bob_ltm.add_session(bob_metrics)
    
    print(f"   Stress: {bob_metrics.stress_duration_ratio*100:.1f}%")
    print(f"   Stability: {bob_metrics.avg_stability*100:.1f}%")
    
    print("\n✓ Data stored in separate files:")
    print(f"   - alice_demo_longterm_memory.json")
    print(f"   - bob_demo_longterm_memory.json")
    print("\n   No cross-contamination!")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("🚀 PHASE 4.1 & 4.2 ENHANCEMENTS DEMO")
    print("="*70)
    print("\nNew Features:")
    print("  ✓ User management and selection")
    print("  ✓ Session limits (max 10/day)")
    print("  ✓ Data export (CSV/JSON)")
    print("  ✓ Archive before deletion")
    print("  ✓ Multi-user data separation")
    
    # Run demos
    alice_id, bob_id, charlie_id = demo_user_management()
    demo_session_limits(alice_id)
    demo_data_export(alice_id)
    demo_archive_functionality(bob_id)
    demo_multi_user_separation()
    
    print("\n" + "="*70)
    print("✅ ALL DEMOS COMPLETE")
    print("="*70)
    print("\nPhase 4.1 & 4.2 features are working!")
    print("\nCheck data/demo_memory/ for created files:")
    print("  - users.json (user registry)")
    print("  - *_longterm_memory.json (per-user data)")
    print("  - *_export_*.csv (exported data)")
    print("  - archive/*.json (archived old data)")


if __name__ == "__main__":
    main()
