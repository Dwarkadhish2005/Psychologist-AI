"""
PHASE 5: PERSONALITY VISUALIZATION
===================================

Create visual representations of Personality State Vector (PSV)

Author: Psychologist AI Team
Date: January 11, 2026
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import json
import sys

# Ensure project root is in path so imports work when run directly
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Try to import PersonalityEngine
try:
    from inference.phase5_personality_engine import PersonalityEngine, PersonalityStateVector
    PHASE5_AVAILABLE = True
except ImportError as e:
    PHASE5_AVAILABLE = False
    PersonalityEngine = None
    PersonalityStateVector = None
    print(f"⚠️ Phase 5 not available for visualization: {e}")


def create_psv_radar_chart(
    psv: 'PersonalityStateVector',
    save_path: Optional[str] = None,
    show_confidence: bool = True,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Create radar chart for PSV traits
    
    Args:
        psv: PersonalityStateVector instance
        save_path: Optional path to save figure
        show_confidence: Whether to show confidence in title
        title: Optional custom title
    
    Returns:
        Matplotlib figure
    """
    # Trait labels (in order)
    labels = [
        'Emotional\nStability',
        'Stress\nSensitivity',
        'Recovery\nSpeed',
        'Positivity\nBias',
        'Volatility'
    ]
    
    # Trait values
    values = [
        psv.emotional_stability,
        psv.stress_sensitivity,
        psv.recovery_speed,
        psv.positivity_bias,
        psv.volatility
    ]
    
    # Number of traits
    n = len(labels)
    
    # Compute angles for radar chart
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    
    # Complete the circle
    values += values[:1]
    angles += angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB', label='Current PSV')
    ax.fill(angles, values, alpha=0.25, color='#2E86AB')
    
    # Fix axis to go in the right order
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12, weight='bold')
    
    # Set y-axis (radial) limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10, color='gray')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set title
    if title is None:
        title = "Personality State Vector (PSV)"
        if show_confidence:
            confidence_pct = int(psv.confidence * 100)
            confidence_level = psv.get_confidence_level().replace('_', ' ').title()
            title += f"\nConfidence: {confidence_level} ({confidence_pct}%)"
    
    ax.set_title(title, size=16, weight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Radar chart saved: {save_path}")
    
    return fig


def create_psv_trend_chart(
    personality_engine: 'PersonalityEngine',
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Create line chart showing PSV trait evolution over time
    
    Args:
        personality_engine: PersonalityEngine instance
        save_path: Optional path to save figure
        title: Optional custom title
    
    Returns:
        Matplotlib figure
    """
    psv = personality_engine.psv
    
    # Get trait histories
    traits = {
        'Emotional Stability': psv.emotional_stability_history,
        'Stress Sensitivity': psv.stress_sensitivity_history,
        'Recovery Speed': psv.recovery_speed_history,
        'Positivity Bias': psv.positivity_bias_history,
        'Volatility': psv.volatility_history
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors for each trait
    colors = {
        'Emotional Stability': '#2E86AB',
        'Stress Sensitivity': '#A23B72',
        'Recovery Speed': '#F18F01',
        'Positivity Bias': '#C73E1D',
        'Volatility': '#6A994E'
    }
    
    # Plot each trait
    for trait_name, history in traits.items():
        if len(history) > 0:
            x = range(len(history))
            ax.plot(x, history, marker='o', linewidth=2, 
                   color=colors[trait_name], label=trait_name)
    
    # Formatting
    ax.set_xlabel('Update Number', size=12, weight='bold')
    ax.set_ylabel('Trait Value (0-1)', size=12, weight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Title
    if title is None:
        title = f"PSV Trait Evolution\n{psv.total_sessions_processed} Sessions Analyzed"
    ax.set_title(title, size=14, weight='bold', pad=15)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Trend chart saved: {save_path}")
    
    return fig


def create_psv_bar_chart(
    psv: 'PersonalityStateVector',
    save_path: Optional[str] = None,
    show_trends: bool = True
) -> plt.Figure:
    """
    Create horizontal bar chart for PSV traits with trend indicators
    
    Args:
        psv: PersonalityStateVector instance
        save_path: Optional path to save figure
        show_trends: Whether to show trend arrows
    
    Returns:
        Matplotlib figure
    """
    # Trait labels and values
    traits = {
        'Emotional Stability': psv.emotional_stability,
        'Stress Sensitivity': psv.stress_sensitivity,
        'Recovery Speed': psv.recovery_speed,
        'Positivity Bias': psv.positivity_bias,
        'Volatility': psv.volatility
    }
    
    # Get trends if requested
    trends = psv.get_trait_trends() if show_trends else {}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    names = list(traits.keys())
    values = list(traits.values())
    
    # Color by value (gradient from red to yellow to green)
    colors = []
    for val in values:
        if val < 0.4:
            colors.append('#E63946')  # Red (low)
        elif val < 0.7:
            colors.append('#F1C40F')  # Yellow (medium)
        else:
            colors.append('#2ECC71')  # Green (high)
    
    # Create horizontal bars
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        # Value text
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', ha='left', fontsize=11, weight='bold')
        
        # Trend arrow if available
        if show_trends and names[i].lower().replace(' ', '_') in trends:
            trend_key = names[i].lower().replace(' ', '_')
            trend = trends[trend_key]
            
            if trend == 'increasing':
                arrow = '↑'
                color = 'green'
            elif trend == 'decreasing':
                arrow = '↓'
                color = 'red'
            else:
                arrow = '→'
                color = 'gray'
            
            ax.text(val + 0.15, bar.get_y() + bar.get_height()/2,
                   arrow, va='center', ha='left', fontsize=16, 
                   color=color, weight='bold')
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11, weight='bold')
    ax.set_xlabel('Trait Value (0-1)', fontsize=12, weight='bold')
    ax.set_xlim(0, 1.3)
    ax.set_title(f"Personality State Vector (PSV)\nConfidence: {psv.get_confidence_level().replace('_', ' ').title()}", 
                fontsize=14, weight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add reference lines
    ax.axvline(x=0.4, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(x=0.7, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Bar chart saved: {save_path}")
    
    return fig


def create_comprehensive_dashboard(
    personality_engine: 'PersonalityEngine',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive PSV dashboard with multiple visualizations
    
    Args:
        personality_engine: PersonalityEngine instance
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    psv = personality_engine.psv
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ===== SUBPLOT 1: Radar Chart =====
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    
    labels = ['Emotional\nStability', 'Stress\nSensitivity', 'Recovery\nSpeed', 
             'Positivity\nBias', 'Volatility']
    values = [psv.emotional_stability, psv.stress_sensitivity, 
             psv.recovery_speed, psv.positivity_bias, psv.volatility]
    
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_plot = values + values[:1]
    angles_plot = angles + angles[:1]
    
    ax1.plot(angles_plot, values_plot, 'o-', linewidth=2, color='#2E86AB')
    ax1.fill(angles_plot, values_plot, alpha=0.25, color='#2E86AB')
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_xticks(angles)
    ax1.set_xticklabels(labels, size=10, weight='bold')
    ax1.set_ylim(0, 1)
    ax1.set_title('PSV Radar', size=12, weight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # ===== SUBPLOT 2: Trend Lines =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    traits = {
        'Emotional Stability': psv.emotional_stability_history,
        'Stress Sensitivity': psv.stress_sensitivity_history,
        'Recovery Speed': psv.recovery_speed_history,
        'Positivity Bias': psv.positivity_bias_history,
        'Volatility': psv.volatility_history
    }
    
    colors_trend = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    for (trait_name, history), color in zip(traits.items(), colors_trend):
        if len(history) > 0:
            ax2.plot(range(len(history)), history, marker='o', 
                    linewidth=2, color=color, label=trait_name, markersize=4)
    
    ax2.set_xlabel('Update Number', size=10, weight='bold')
    ax2.set_ylabel('Value', size=10, weight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=8)
    ax2.set_title('Trait Evolution', size=12, weight='bold')
    
    # ===== SUBPLOT 3: Bar Chart =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    trait_names = ['Emotional\nStability', 'Stress\nSensitivity', 'Recovery\nSpeed', 
                   'Positivity\nBias', 'Volatility']
    trait_values = [psv.emotional_stability, psv.stress_sensitivity, 
                   psv.recovery_speed, psv.positivity_bias, psv.volatility]
    
    colors_bar = ['#2ECC71' if v >= 0.7 else '#F1C40F' if v >= 0.4 else '#E63946' 
                  for v in trait_values]
    
    bars = ax3.barh(range(len(trait_names)), trait_values, 
                   color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax3.set_yticks(range(len(trait_names)))
    ax3.set_yticklabels(trait_names, fontsize=9, weight='bold')
    ax3.set_xlabel('Value', fontsize=10, weight='bold')
    ax3.set_xlim(0, 1.1)
    ax3.set_title('Current Values', size=12, weight='bold')
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars, trait_values):
        ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9, weight='bold')
    
    # ===== SUBPLOT 4: Metadata =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Get behavioral descriptor
    descriptor = psv.get_behavioral_descriptor()
    
    # Metadata text
    metadata_text = f"""
PERSONALITY ASSESSMENT

User ID: {personality_engine.user_id}
Last Updated: {psv.last_updated[:19]}

CONFIDENCE
  Level: {psv.get_confidence_level().replace('_', ' ').title()}
  Score: {psv.confidence:.1%}
  Sessions: {psv.total_sessions_processed}

CURRENT TRENDS
  Emotional Stability: {psv.get_trait_trends()['emotional_stability']}
  Stress Sensitivity: {psv.get_trait_trends()['stress_sensitivity']}
  Recovery Speed: {psv.get_trait_trends()['recovery_speed']}
  Positivity Bias: {psv.get_trait_trends()['positivity_bias']}
  Volatility: {psv.get_trait_trends()['volatility']}

BEHAVIORAL DESCRIPTOR
{descriptor}

ETHICAL NOTICE
This is a behavioral pattern assessment,
not a clinical diagnosis.
    """.strip()
    
    ax4.text(0.05, 0.95, metadata_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Main title
    fig.suptitle('Personality State Vector (PSV) - Comprehensive Dashboard', 
                fontsize=16, weight='bold', y=0.98)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard saved: {save_path}")
    
    return fig


# ============================================================================
# DEMO
# ============================================================================


if __name__ == "__main__":
    if not PHASE5_AVAILABLE:
        print("❌ Phase 5 not available. Cannot create visualizations.")
        exit(1)

    from datetime import datetime

    print("📊 Personality Visualization Report Generator")
    print("=" * 70)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    storage_dir = Path("data/user_memory")
    output_dir = Path("assets/reports/psv_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all users that have PSV files on disk
    psv_files = sorted(storage_dir.glob("*_psv.json"))

    if not psv_files:
        print("❌ No PSV files found in data/user_memory/")
        print("   Run the integrated system and complete at least 3 sessions first.")
        exit(1)

    print(f"Found {len(psv_files)} user(s) with personality data:\n")

    total_generated = 0
    for psv_file in psv_files:
        # Derive user_id from filename: strip the trailing _psv.json
        user_id = psv_file.stem.replace("_psv", "")

        print(f"─── User: {user_id} ───")

        # Load personality engine (reads from disk)
        engine = PersonalityEngine(user_id=user_id, storage_dir=str(storage_dir))

        sessions = engine.psv.total_sessions_processed
        confidence = engine.psv.confidence
        last_updated = engine.psv.last_updated[:19]

        print(f"    Sessions: {sessions}  |  Confidence: {confidence:.1%}  |  Last updated: {last_updated}")

        if not engine.can_infer_personality():
            needed = engine.min_sessions_required - sessions
            print(f"    ⏳ Skipped — need {needed} more session(s) to unlock personality reports\n")
            continue

        # Generate all 4 charts, overwriting any existing files
        try:
            create_psv_radar_chart(
                engine.psv,
                save_path=str(output_dir / f"{user_id}_radar.png")
            )
            create_psv_trend_chart(
                engine,
                save_path=str(output_dir / f"{user_id}_trends.png")
            )
            create_psv_bar_chart(
                engine.psv,
                save_path=str(output_dir / f"{user_id}_bars.png")
            )
            create_comprehensive_dashboard(
                engine,
                save_path=str(output_dir / f"{user_id}_dashboard.png")
            )
            plt.close('all')  # Free memory
            total_generated += 4
            print()
        except Exception as e:
            print(f"    ❌ Error generating charts: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 70)
    print(f"✓ Done — {total_generated} chart(s) saved to: {output_dir}")
    print("=" * 70)

