"""
Quick test script for Phase 5 visualization
"""
import sys
from pathlib import Path

# Ensure proper path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.phase5_personality_engine import PersonalityEngine
from inference.phase5_visualization import (
    create_psv_radar_chart,
    create_psv_trend_chart,
    create_psv_bar_chart,
    create_comprehensive_dashboard
)
from inference.phase4_user_manager import UserManager

print("🎨 Phase 5 Visualization Test")
print("=" * 70)

# Get user
user_manager = UserManager()
users = user_manager.list_users()

if not users:
    print("❌ No users found. Run the integrated system first.")
    exit(1)

# Check all users and find one with enough sessions
print(f"Found {len(users)} user(s) in system:")
for u in users:
    engine_temp = PersonalityEngine(user_id=u.user_id, storage_dir="data/user_memory")
    can_infer = engine_temp.can_infer_personality()
    status = "✓ Ready" if can_infer else f"✗ Needs {engine_temp.min_sessions_required - u.total_sessions} more"
    print(f"  • {u.name}: {u.total_sessions} sessions - {status}")
print()

# Find first user with enough sessions
user = None
for u in users:
    engine_temp = PersonalityEngine(user_id=u.user_id, storage_dir="data/user_memory")
    if engine_temp.can_infer_personality():
        user = u
        break

if user is None:
    print("❌ No users have enough sessions for personality inference")
    print(f"   Minimum required: 3 sessions")
    print()
    print("💡 Solution: Run the psychologist AI more times with your users")
    exit(1)

print(f"✓ Using user: {user.name}")
print(f"  User ID: {user.user_id}")
print(f"  Total sessions: {user.total_sessions}")
print()

# Load personality engine
print("Loading personality engine...")
engine = PersonalityEngine(user_id=user.user_id, storage_dir="data/user_memory")

if not engine.can_infer_personality():
    print(f"❌ User needs {engine.min_sessions_required} sessions minimum")
    print(f"   Current sessions: {engine.psv.total_sessions_processed}")
    exit(1)

print(f"✓ Personality engine loaded")
print(f"  Sessions processed: {engine.psv.total_sessions_processed}")
print(f"  Confidence: {engine.psv.confidence:.1%}")
print()

# Show PSV summary
summary = engine.get_personality_summary()
psv = summary['personality_state_vector']

print("📊 Personality State Vector (PSV):")
print(f"  • Emotional Stability:  {psv['emotional_stability']:.3f}")
print(f"  • Stress Sensitivity:   {psv['stress_sensitivity']:.3f}")
print(f"  • Recovery Speed:       {psv['recovery_speed']:.3f}")
print(f"  • Positivity Bias:      {psv['positivity_bias']:.3f}")
print(f"  • Volatility:           {psv['volatility']:.3f}")
print()

print("📝 Behavioral Descriptor:")
print(f"  {summary['behavioral_descriptor']}")
print()

# Create output directory
output_dir = Path("assets/reports/psv_visualizations")
output_dir.mkdir(parents=True, exist_ok=True)

print("🎨 Generating visualizations...")
print()

# 1. Radar chart
print("  1. Creating radar chart...")
try:
    create_psv_radar_chart(
        engine.psv,
        save_path=str(output_dir / f"{user.user_id}_radar.png")
    )
    print("     ✓ Saved")
except Exception as e:
    print(f"     ✗ Error: {e}")

# 2. Trend chart (only if history exists)
if len(engine.psv.emotional_stability_history) > 1:
    print("  2. Creating trend chart...")
    try:
        create_psv_trend_chart(
            engine,
            save_path=str(output_dir / f"{user.user_id}_trends.png")
        )
        print("     ✓ Saved")
    except Exception as e:
        print(f"     ✗ Error: {e}")
else:
    print("  2. Skipping trend chart (need more updates)")

# 3. Bar chart
print("  3. Creating bar chart...")
try:
    create_psv_bar_chart(
        engine.psv,
        save_path=str(output_dir / f"{user.user_id}_bars.png")
    )
    print("     ✓ Saved")
except Exception as e:
    print(f"     ✗ Error: {e}")

# 4. Comprehensive dashboard
print("  4. Creating comprehensive dashboard...")
try:
    create_comprehensive_dashboard(
        engine,
        save_path=str(output_dir / f"{user.user_id}_dashboard.png")
    )
    print("     ✓ Saved")
except Exception as e:
    print(f"     ✗ Error: {e}")

print()
print("=" * 70)
print(f"✅ Visualizations saved to: {output_dir}")
print("=" * 70)
