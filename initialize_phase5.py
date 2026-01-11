"""
Initialize Phase 5 PSV from existing Phase 4 data
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.phase5_personality_engine import PersonalityEngine
from inference.phase4_cognitive_layer import LongTermMemory
from inference.phase4_user_manager import UserManager

print("🔄 Phase 5 PSV Initialization from Existing Data")
print("=" * 70)

# Get user
user_manager = UserManager()
users = user_manager.list_users()

if not users:
    print("❌ No users found")
    exit(1)

user = users[0]
print(f"✓ User: {user.name} ({user.user_id})")
print(f"  Total sessions: {user.total_sessions}")
print()

# Load long-term memory
print("Loading Phase 4 long-term memory...")
ltm = LongTermMemory(user_id=user.user_id, storage_dir="data/user_memory")

daily_profiles = list(ltm.daily_profiles.values())
print(f"✓ Found {len(daily_profiles)} daily profiles")

if not daily_profiles:
    print("❌ No daily profiles found")
    exit(1)

# Show daily profile dates
for profile in sorted(daily_profiles, key=lambda p: p.date):
    print(f"  - {profile.date}: {profile.total_sessions} session(s)")
print()

# Initialize Phase 5 engine
print("Initializing Phase 5 personality engine...")
engine = PersonalityEngine(
    user_id=user.user_id,
    storage_dir="data/user_memory",
    learning_rate=0.03,
    min_sessions_required=3
)

print(f"✓ Engine created")
print(f"  Current PSV sessions: {engine.psv.total_sessions_processed}")
print()

# If PSV is empty, we need to simulate adding sessions
if engine.psv.total_sessions_processed == 0:
    print("PSV is empty - need to update from Phase 4 data")
    print()
    
    # Get recent daily profiles (last 7 days)
    recent_dates = sorted(ltm.daily_profiles.keys())[-7:]
    recent_profiles = [ltm.daily_profiles[d] for d in recent_dates]
    
    print(f"Using {len(recent_profiles)} recent daily profiles for PSV update...")
    
    # Manually set session count to match reality
    engine.psv.total_sessions_processed = sum(p.total_sessions for p in daily_profiles)
    
    print(f"✓ Set session count: {engine.psv.total_sessions_processed}")
    
    # Now update PSV
    if engine.can_infer_personality():
        print("✓ Enough data - updating PSV...")
        engine.update_psv(recent_profiles)
        
        print()
        print("📊 PSV Updated Successfully!")
        print(f"  • Emotional Stability:  {engine.psv.emotional_stability:.3f}")
        print(f"  • Stress Sensitivity:   {engine.psv.stress_sensitivity:.3f}")
        print(f"  • Recovery Speed:       {engine.psv.recovery_speed:.3f}")
        print(f"  • Positivity Bias:      {engine.psv.positivity_bias:.3f}")
        print(f"  • Volatility:           {engine.psv.volatility:.3f}")
        print()
        print(f"  Confidence: {engine.psv.confidence:.1%} ({engine.psv.get_confidence_level()})")
        print()
        
        # Generate behavioral descriptor
        summary = engine.get_personality_summary()
        print("📝 Behavioral Descriptor:")
        print(f"  {summary['behavioral_descriptor']}")
        print()
        
    else:
        print(f"❌ Still need {engine.min_sessions_required} sessions")

else:
    print(f"✓ PSV already initialized with {engine.psv.total_sessions_processed} sessions")
    summary = engine.get_personality_summary()
    psv = summary['personality_state_vector']
    
    print()
    print("📊 Current PSV:")
    print(f"  • Emotional Stability:  {psv['emotional_stability']:.3f}")
    print(f"  • Stress Sensitivity:   {psv['stress_sensitivity']:.3f}")
    print(f"  • Recovery Speed:       {psv['recovery_speed']:.3f}")
    print(f"  • Positivity Bias:      {psv['positivity_bias']:.3f}")
    print(f"  • Volatility:           {psv['volatility']:.3f}")
    print()

print("=" * 70)
print("✅ Phase 5 PSV initialization complete!")
print("=" * 70)
