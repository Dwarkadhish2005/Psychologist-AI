"""
PHASE 4 INTEGRATION DEMO
========================

Demonstrates Phase 4 working with Phase 3 output.
Shows personality inference, deviation detection, and risk adjustment.

Usage:
    python inference/demo_phase4_integration.py
"""

import sys
from pathlib import Path
import time
import random

sys.path.append(str(Path(__file__).parent.parent))

from inference.phase3_multimodal_fusion import (
    PsychologicalState,
    MentalState,
    RiskLevel
)
from inference.phase4_cognitive_layer import Phase4CognitiveFusion


def simulate_psychological_states(phase4, scenario="normal"):
    """
    Simulate a series of psychological states
    
    Args:
        phase4: Phase4CognitiveFusion instance
        scenario: "normal", "stressed", or "deteriorating"
    """
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario.upper()}")
    print(f"{'='*80}\n")
    
    # Define state patterns for each scenario
    if scenario == "normal":
        states = [
            (MentalState.CALM, RiskLevel.LOW, 0.8, 0.8, "happy", None),
            (MentalState.CALM, RiskLevel.LOW, 0.75, 0.85, "neutral", None),
            (MentalState.STABLE_POSITIVE, RiskLevel.LOW, 0.82, 0.8, "happy", None),
            (MentalState.CALM, RiskLevel.LOW, 0.78, 0.82, "neutral", None),
            (MentalState.JOYFUL, RiskLevel.LOW, 0.85, 0.9, "happy", None),
        ]
        
    elif scenario == "stressed":
        states = [
            (MentalState.STRESSED, RiskLevel.MODERATE, 0.7, 0.6, "neutral", None),
            (MentalState.ANXIOUS, RiskLevel.MODERATE, 0.65, 0.55, "sad", None),
            (MentalState.STRESSED, RiskLevel.HIGH, 0.68, 0.5, "fear", "sad"),
            (MentalState.OVERWHELMED, RiskLevel.HIGH, 0.6, 0.45, "fear", "sad"),
            (MentalState.STRESSED, RiskLevel.HIGH, 0.62, 0.48, "angry", None),
        ]
        
    else:  # deteriorating
        states = [
            (MentalState.CALM, RiskLevel.LOW, 0.8, 0.8, "neutral", None),
            (MentalState.STRESSED, RiskLevel.MODERATE, 0.72, 0.7, "neutral", None),
            (MentalState.ANXIOUS, RiskLevel.MODERATE, 0.65, 0.6, "sad", None),
            (MentalState.OVERWHELMED, RiskLevel.HIGH, 0.58, 0.5, "fear", "sad"),
            (MentalState.EMOTIONALLY_UNSTABLE, RiskLevel.CRITICAL, 0.5, 0.3, "fear", "sad"),
        ]
    
    # Process each state
    for i, (mental_state, risk_level, confidence, stability, emotion, hidden) in enumerate(states, 1):
        # Create PsychologicalState
        state = PsychologicalState(
            dominant_emotion=emotion,
            hidden_emotion=hidden,
            mental_state=mental_state,
            risk_level=risk_level,
            confidence=confidence,
            stability_score=stability,
            explanations=[f"Simulated {scenario} state {i}"],
            temporal_patterns=[],
            raw_signals={},
            timestamp=time.time()
        )
        
        # Process through Phase 4
        profile = phase4.process_state(state)
        
        # Display results
        print(f"\n--- State {i}/5 ---")
        print(f"Mental State: {mental_state.value}")
        print(f"Phase 3 Risk: {risk_level.value}")
        print(f"Phase 4 Risk: {profile.adjusted_risk.value}")
        if profile.adjusted_risk != risk_level:
            print(f"  → {profile.risk_adjustment_reason}")
        
        if profile.deviations:
            print(f"⚠️  Deviations detected: {len(profile.deviations)}")
            for dev in profile.deviations[:2]:
                print(f"   • {dev.deviation_type}: {dev.severity:.0%}")
        
        time.sleep(0.1)  # Small delay for realism


def main():
    """Main demo"""
    print("="*80)
    print("🧠 PHASE 4 INTEGRATION DEMO")
    print("="*80)
    print("\nInitializing Phase 4 Cognitive Layer...")
    
    # Initialize Phase 4
    phase4 = Phase4CognitiveFusion(
        user_id="demo_user",
        storage_dir="data/demo_memory"
    )
    
    print("✓ Phase 4 initialized")
    print("\nThis demo simulates 3 scenarios:")
    print("  1. Normal behavior (baseline establishment)")
    print("  2. Stressed behavior (moderate deviations)")
    print("  3. Deteriorating behavior (severe deviations)")
    print("\n" + "="*80)
    
    # Scenario 1: Normal (establish baseline)
    print("\n\n🟢 SCENARIO 1: NORMAL BEHAVIOR")
    print("Purpose: Establish baseline personality and normal patterns")
    input("\nPress Enter to start...")
    
    for session in range(3):
        print(f"\n--- Session {session + 1}/3 ---")
        for _ in range(10):  # 10 states per session
            simulate_psychological_states(phase4, "normal")
        
        # End session
        if phase4.session_memory.get_frame_count() > 0:
            metrics = phase4.session_memory.reset()
            phase4.long_term_memory.add_session(metrics)
            print(f"\n✓ Session {session + 1} saved to long-term memory")
    
    # Update profiles
    phase4._update_profiles()
    
    print("\n\n📊 BASELINE ESTABLISHED")
    print(f"  • Personality confidence: {phase4.personality.confidence:.0%}")
    print(f"  • Baseline confidence: {phase4.baseline.confidence:.0%}")
    print(f"  • Data: {phase4.personality.data_days} days")
    
    # Scenario 2: Stressed behavior
    print("\n\n🟡 SCENARIO 2: STRESSED BEHAVIOR")
    print("Purpose: Test deviation detection with moderate stress")
    input("\nPress Enter to continue...")
    
    simulate_psychological_states(phase4, "stressed")
    
    # Scenario 3: Deteriorating behavior
    print("\n\n🔴 SCENARIO 3: DETERIORATING BEHAVIOR")
    print("Purpose: Test severe deviation detection and risk escalation")
    input("\nPress Enter to continue...")
    
    simulate_psychological_states(phase4, "deteriorating")
    
    # Final summary
    print("\n\n" + "="*80)
    print("📋 FINAL PHASE 4 PROFILE SUMMARY")
    print("="*80)
    
    # Get latest profile
    test_state = PsychologicalState(
        dominant_emotion="neutral",
        hidden_emotion=None,
        mental_state=MentalState.CALM,
        risk_level=RiskLevel.LOW,
        confidence=0.8,
        stability_score=0.8,
        explanations=["Summary state"],
        temporal_patterns=[],
        raw_signals={},
        timestamp=time.time()
    )
    
    final_profile = phase4.process_state(test_state)
    
    # Print full report
    report = phase4.get_full_report(final_profile)
    print(report)
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE!")
    print("="*80)
    print("\n🎯 Key Takeaways:")
    print("  • Phase 4 builds personality over time")
    print("  • Deviations detected when behavior is unusual FOR THIS USER")
    print("  • Risk adjusted based on personal baseline")
    print("  • Pure statistics - fully explainable")
    print("\n🚀 Ready to integrate with real-time system!")
    print("="*80)


if __name__ == "__main__":
    main()
