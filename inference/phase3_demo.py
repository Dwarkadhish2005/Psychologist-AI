"""
PHASE 3 VISUALIZATION & DEMONSTRATION
=====================================
Interactive demonstration of Phase 3 capabilities.

Usage:
    python inference/phase3_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.phase3_multimodal_fusion import (
    Phase3MultiModalFusion, 
    MentalState, 
    RiskLevel,
    format_psychological_state
)


def visualize_mental_states():
    """Visualize all mental states with risk levels"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Title
    ax.text(5, 15, 'PSYCHOLOGICAL STATE TAXONOMY', 
           ha='center', va='top', fontsize=18, fontweight='bold')
    
    # Mental states grouped by risk
    states_by_risk = {
        RiskLevel.LOW: [
            (MentalState.CALM, "Relaxed, stable"),
            (MentalState.JOYFUL, "Genuinely happy"),
            (MentalState.STABLE_POSITIVE, "Content")
        ],
        RiskLevel.MODERATE: [
            (MentalState.STRESSED, "Under pressure"),
            (MentalState.EMOTIONALLY_MASKED, "Hiding emotions"),
            (MentalState.EMOTIONALLY_UNSTABLE, "Mood swings"),
            (MentalState.FEARFUL, "Afraid"),
            (MentalState.STABLE_NEGATIVE, "Persistent negativity"),
            (MentalState.CONFUSED, "Uncertain")
        ],
        RiskLevel.HIGH: [
            (MentalState.HAPPY_UNDER_STRESS, "Masking stress"),
            (MentalState.ANGRY_STRESSED, "Anger + pressure"),
            (MentalState.SAD_DEPRESSED, "Deep sadness")
        ],
        RiskLevel.CRITICAL: [
            (MentalState.ANXIOUS, "Fear + high stress"),
            (MentalState.OVERWHELMED, "Too much stress"),
            (MentalState.EMOTIONALLY_FLAT, "Disengaged")
        ]
    }
    
    colors = {
        RiskLevel.LOW: '#90EE90',
        RiskLevel.MODERATE: '#FFD700',
        RiskLevel.HIGH: '#FFA500',
        RiskLevel.CRITICAL: '#FF6B6B'
    }
    
    y_pos = 13.5
    for risk, states in states_by_risk.items():
        # Risk level header
        ax.add_patch(Rectangle((0.5, y_pos - 0.4), 9, 0.6, 
                               facecolor=colors[risk], alpha=0.3))
        ax.text(1, y_pos, f'{risk.value.upper()} RISK', 
               fontweight='bold', fontsize=11)
        y_pos -= 1
        
        # States
        for state, description in states:
            ax.add_patch(FancyBboxPatch((1, y_pos - 0.3), 7, 0.5,
                                       boxstyle="round,pad=0.05",
                                       facecolor=colors[risk], 
                                       alpha=0.5, edgecolor='black'))
            ax.text(1.2, y_pos, state.value.replace('_', ' ').title(),
                   fontweight='bold', fontsize=9)
            ax.text(8.2, y_pos, description, fontsize=8, style='italic')
            y_pos -= 0.6
        
        y_pos -= 0.4
    
    plt.tight_layout()
    plt.savefig('phase3_mental_states.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: phase3_mental_states.png")
    plt.show()


def visualize_fusion_architecture():
    """Visualize Phase 3 architecture"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'PHASE 3 ARCHITECTURE', 
           ha='center', fontsize=16, fontweight='bold')
    
    # Layers
    layers = [
        (10, "RAW INPUTS", "Face emotion + Voice emotion + Stress", '#E3F2FD'),
        (8.5, "Layer 1: Signal Normalization", "Reliability weighting + Quality assessment", '#BBDEFB'),
        (7, "Layer 2: Temporal Reasoning", "Memory window + Pattern detection", '#90CAF9'),
        (5.5, "Layer 3: Fusion Logic", "Rule-based psychology + Conflict resolution", '#64B5F6'),
        (4, "Layer 4: Psychological Reasoning", "Mental state inference + Risk assessment", '#42A5F5'),
        (2.5, "OUTPUT", "Psychological State + Explanations", '#2196F3')
    ]
    
    for y, title, desc, color in layers:
        # Box
        ax.add_patch(FancyBboxPatch((1, y - 0.5), 8, 1,
                                   boxstyle="round,pad=0.1",
                                   facecolor=color, edgecolor='black', linewidth=2))
        # Title
        ax.text(5, y + 0.15, title, ha='center', fontweight='bold', fontsize=11)
        # Description
        ax.text(5, y - 0.2, desc, ha='center', fontsize=8, style='italic')
        
        # Arrow
        if y > 2.5:
            ax.annotate('', xy=(5, y - 0.6), xytext=(5, y - 1.3),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    plt.tight_layout()
    plt.savefig('phase3_architecture.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: phase3_architecture.png")
    plt.show()


def demonstrate_scenarios():
    """Run and visualize all test scenarios"""
    
    print("\n" + "=" * 70)
    print("PHASE 3 SCENARIO DEMONSTRATIONS")
    print("=" * 70)
    
    scenarios = [
        {
            'name': "Happy under stress (Emotional Masking)",
            'description': "Person smiling at work but stressed about deadline",
            'inputs': [
                ('happy', 0.75, True, 'happy', 0.65, 0.9, 'high', 0.82)
            ] * 15,
            'expected': "HAPPY_UNDER_STRESS"
        },
        {
            'name': "Emotional Masking",
            'description': "Neutral face hiding fearful voice",
            'inputs': [
                ('neutral', 0.70, True, 'fear', 0.68, 0.85, 'medium', 0.60)
            ] * 15,
            'expected': "EMOTIONALLY_MASKED"
        },
        {
            'name': "Emotional Instability",
            'description': "Rapid mood swings",
            'inputs': [
                ('happy', 0.60, True, 'happy', 0.55, 0.8, 'medium', 0.65),
                ('sad', 0.60, True, 'sad', 0.55, 0.8, 'medium', 0.65),
                ('angry', 0.60, True, 'angry', 0.55, 0.8, 'medium', 0.65),
                ('neutral', 0.60, True, 'neutral', 0.55, 0.8, 'medium', 0.65),
                ('fear', 0.60, True, 'fear', 0.55, 0.8, 'medium', 0.65),
            ] * 3,
            'expected': "EMOTIONALLY_UNSTABLE"
        },
        {
            'name': "Calm and Stable",
            'description': "Consistent neutral emotion, low stress",
            'inputs': [
                ('neutral', 0.80, True, 'neutral', 0.75, 0.95, 'low', 0.85)
            ] * 15,
            'expected': "CALM"
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n{'=' * 70}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Expected: {scenario['expected']}")
        print('-' * 70)
        
        phase3 = Phase3MultiModalFusion()
        
        # Process inputs
        for inputs in scenario['inputs']:
            state = phase3.process_frame(*inputs)
        
        # Print result
        print(f"\n{format_psychological_state(state)}")
        
        # Check if matches expected
        match = state.mental_state.value.upper() == scenario['expected'].upper()
        status = "✅ PASS" if match else "❌ FAIL"
        print(f"\nResult: {status}")
        
        results.append({
            'scenario': scenario['name'],
            'expected': scenario['expected'],
            'actual': state.mental_state.value,
            'match': match,
            'risk': state.risk_level.value,
            'confidence': state.confidence,
            'stability': state.stability_score
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r['match'])
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    for result in results:
        status = "✅" if result['match'] else "❌"
        print(f"\n{status} {result['scenario']}")
        print(f"   Expected: {result['expected']}")
        print(f"   Got: {result['actual']}")
        print(f"   Risk: {result['risk']} | Confidence: {result['confidence']*100:.1f}% | Stability: {result['stability']*100:.0f}%")
    
    return results


def visualize_scenario_comparison(results):
    """Visualize scenario results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PHASE 3 SCENARIO RESULTS', fontsize=16, fontweight='bold')
    
    scenarios = [r['scenario'] for r in results]
    
    # Risk levels
    ax = axes[0, 0]
    risk_colors = {'low': '#90EE90', 'moderate': '#FFD700', 
                   'high': '#FFA500', 'critical': '#FF6B6B'}
    colors = [risk_colors[r['risk']] for r in results]
    ax.bar(range(len(scenarios)), [1]*len(scenarios), color=colors)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.split()[0] for s in scenarios], rotation=45, ha='right')
    ax.set_yticks([])
    ax.set_title('Risk Levels', fontweight='bold')
    ax.set_ylim(0, 1.2)
    
    # Add legend
    legend_patches = [mpatches.Patch(color=color, label=level.upper()) 
                     for level, color in risk_colors.items()]
    ax.legend(handles=legend_patches, loc='upper right')
    
    # Confidence
    ax = axes[0, 1]
    confidences = [r['confidence']*100 for r in results]
    bars = ax.bar(range(len(scenarios)), confidences, color='#64B5F6')
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.split()[0] for s in scenarios], rotation=45, ha='right')
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Confidence Scores', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, confidences)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Stability
    ax = axes[1, 0]
    stabilities = [r['stability']*100 for r in results]
    bars = ax.bar(range(len(scenarios)), stabilities, color='#90CAF9')
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.split()[0] for s in scenarios], rotation=45, ha='right')
    ax.set_ylabel('Stability (%)')
    ax.set_title('Emotional Stability', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(70, color='green', linestyle='--', alpha=0.5, label='Stable threshold')
    ax.legend()
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, stabilities)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # Test results
    ax = axes[1, 1]
    matches = [1 if r['match'] else 0 for r in results]
    colors = ['#90EE90' if m else '#FF6B6B' for m in matches]
    ax.bar(range(len(scenarios)), matches, color=colors)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.split()[0] for s in scenarios], rotation=45, ha='right')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['FAIL', 'PASS'])
    ax.set_title('Test Results', fontweight='bold')
    ax.set_ylim(-0.1, 1.2)
    
    # Add status text
    for i, (match, scenario) in enumerate(zip(matches, scenarios)):
        status = "✅ PASS" if match else "❌ FAIL"
        ax.text(i, match + 0.05, status, ha='center', va='bottom', 
               fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('phase3_scenario_results.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved: phase3_scenario_results.png")
    plt.show()


def main():
    """Main demonstration"""
    
    print("\n" + "=" * 70)
    print("PHASE 3 DEMONSTRATION & VISUALIZATION")
    print("=" * 70)
    
    # 1. Visualize mental states
    print("\n1. Generating mental states taxonomy...")
    visualize_mental_states()
    
    # 2. Visualize architecture
    print("\n2. Generating architecture diagram...")
    visualize_fusion_architecture()
    
    # 3. Run scenarios
    print("\n3. Running scenario demonstrations...")
    results = demonstrate_scenarios()
    
    # 4. Visualize results
    print("\n4. Generating scenario comparison...")
    visualize_scenario_comparison(results)
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • phase3_mental_states.png - Mental state taxonomy")
    print("  • phase3_architecture.png - System architecture")
    print("  • phase3_scenario_results.png - Test results comparison")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
