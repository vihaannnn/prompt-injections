"""
LLM Security Research: Prompt Injection Defense Statistical Analysis
=====================================================================

This script performs comprehensive statistical analysis comparing defense 
effectiveness between two scenarios:
- Systems WITH access codes in system prompts
- Systems WITHOUT access codes in system prompts

Includes:
- Cohen's h effect size calculation
- 95% confidence intervals using Wilson score interval
- McNemar's test for paired data
- Multi-panel visualization
- Detailed text report


Date: 2025-11-19 at 11:15 AM
https://claude.ai/share/767df928-5521-4792-ba91-ca4a6d20dda8
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# REAL PROJECT DATA
# =============================================================================

# From: D:\LLM_judge\results\defense_evaluation_filtered\defense_evaluation_report_filtered.txt

REAL_DATA = {
    'with_access_code': {
        'total_dataset': 100,
        'evaluated': 38,
        'skipped': 62,
        'original_successful': 35,
        'original_failed': 3,
        'improved_successful': 27,
        'improved_failed': 11,
        'newly_blocked': 11,
        'newly_successful': 3,
    },
    'without_access_code': {
        'total_dataset': 100,
        'evaluated': 20,
        'skipped': 80,
        'original_successful': 18,
        'original_failed': 2,
        'improved_successful': 10,
        'improved_failed': 10,
        'newly_blocked': 9,
        'newly_successful': 1,
    }
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ScenarioData:
    """Container for scenario evaluation data"""
    name: str
    total_dataset: int
    evaluated: int
    skipped: int
    original_successful: int
    original_failed: int
    improved_successful: int
    improved_failed: int
    newly_blocked: int
    newly_successful: int
    
    @property
    def original_success_rate(self) -> float:
        return self.original_successful / self.evaluated
    
    @property
    def improved_success_rate(self) -> float:
        return self.improved_successful / self.evaluated
    
    @property
    def success_rate_reduction(self) -> float:
        return self.original_success_rate - self.improved_success_rate


@dataclass
class StatisticalResults:
    """Container for all statistical analysis results"""
    scenario: ScenarioData
    
    # Effect Size
    cohens_h: float = 0.0
    cohens_h_interpretation: str = ""
    
    # Confidence Intervals (Wilson score)
    ci_original: Tuple[float, float] = (0.0, 0.0)
    ci_improved: Tuple[float, float] = (0.0, 0.0)
    ci_reduction: Tuple[float, float] = (0.0, 0.0)
    
    # McNemar's Test
    mcnemar_chi2: float = 0.0
    mcnemar_p_value: float = 0.0
    is_significant: bool = False
    
    # Additional metrics
    odds_ratio: float = 0.0
    relative_risk: float = 0.0


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def cohens_h(p1: float, p2: float) -> float:
    """
    Calculate Cohen's h effect size for comparing two proportions.
    
    Formula: h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    
    Parameters:
    -----------
    p1 : float
        First proportion (e.g., improved success rate)
    p2 : float
        Second proportion (e.g., original success rate)
    
    Returns:
    --------
    float : Cohen's h value
    
    Interpretation:
    ---------------
    |h| < 0.2  : Small effect
    |h| = 0.5  : Medium effect
    |h| > 0.8  : Large effect
    """
    # Handle edge cases
    p1 = np.clip(p1, 0.0001, 0.9999)
    p2 = np.clip(p2, 0.0001, 0.9999)
    
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    
    return phi1 - phi2


def interpret_cohens_h(h: float) -> str:
    """Interpret Cohen's h effect size magnitude"""
    abs_h = abs(h)
    if abs_h < 0.2:
        return "Small"
    elif abs_h < 0.5:
        return "Small-to-Medium"
    elif abs_h < 0.8:
        return "Medium"
    elif abs_h < 1.2:
        return "Large"
    else:
        return "Very Large"


def wilson_score_interval(successes: int, total: int, 
                          confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a proportion.
    
    More accurate than normal approximation, especially for:
    - Small sample sizes
    - Proportions near 0 or 1
    
    Parameters:
    -----------
    successes : int
        Number of successes
    total : int
        Total number of trials
    confidence : float
        Confidence level (default 0.95 for 95% CI)
    
    Returns:
    --------
    Tuple[float, float] : (lower_bound, upper_bound)
    """
    if total == 0:
        return (0.0, 0.0)
    
    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    offset = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    
    lower = max(0.0, (centre - offset) / denominator)
    upper = min(1.0, (centre + offset) / denominator)
    
    return (lower, upper)


def newcombe_wilson_ci(p1: float, n1: int, p2: float, n2: int,
                       confidence: float = 0.95) -> Tuple[float, float]:
    """
    Newcombe-Wilson confidence interval for difference between proportions.
    
    Uses Wilson score intervals for each proportion and combines them.
    
    Parameters:
    -----------
    p1, n1 : First proportion and sample size
    p2, n2 : Second proportion and sample size
    confidence : Confidence level
    
    Returns:
    --------
    Tuple[float, float] : CI for (p1 - p2)
    """
    # Get Wilson CIs for each proportion
    l1, u1 = wilson_score_interval(int(p1 * n1), n1, confidence)
    l2, u2 = wilson_score_interval(int(p2 * n2), n2, confidence)
    
    diff = p1 - p2
    
    # Newcombe method
    lower = diff - np.sqrt((p1 - l1)**2 + (u2 - p2)**2)
    upper = diff + np.sqrt((u1 - p1)**2 + (p2 - l2)**2)
    
    return (lower, upper)


def mcnemar_test(b: int, c: int, correction: bool = True) -> Tuple[float, float]:
    """
    McNemar's test for paired nominal data.
    
    Tests whether the row and column marginal frequencies are equal.
    Used for before-after studies on the same subjects.
    
    Contingency table:
                    After: Success    After: Fail
    Before: Success      a               b (blocked)
    Before: Fail         c (new success) d
    
    Parameters:
    -----------
    b : int
        Cases that were successful before, failed after (blocked by defense)
    c : int
        Cases that failed before, successful after (defense made worse)
    correction : bool
        Use continuity correction (Edwards' correction)
    
    Returns:
    --------
    Tuple[float, float] : (chi-square statistic, p-value)
    """
    if b + c == 0:
        return (0.0, 1.0)
    
    if correction:
        # McNemar's test with continuity correction
        chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
    else:
        chi2 = ((b - c) ** 2) / (b + c)
    
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return (chi2, p_value)


def calculate_odds_ratio(b: int, c: int) -> float:
    """Calculate odds ratio for McNemar's test"""
    if c == 0:
        return float('inf') if b > 0 else 1.0
    return b / c


def calculate_relative_risk(original_rate: float, improved_rate: float) -> float:
    """Calculate relative risk (risk ratio)"""
    if original_rate == 0:
        return 0.0
    return improved_rate / original_rate


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def perform_statistical_analysis(data: ScenarioData) -> StatisticalResults:
    """
    Perform complete statistical analysis for a scenario.
    
    Parameters:
    -----------
    data : ScenarioData
        Scenario evaluation data
    
    Returns:
    --------
    StatisticalResults : Complete analysis results
    """
    results = StatisticalResults(scenario=data)
    
    # 1. Cohen's h Effect Size
    results.cohens_h = cohens_h(
        data.improved_success_rate,
        data.original_success_rate
    )
    results.cohens_h_interpretation = interpret_cohens_h(results.cohens_h)
    
    # 2. Wilson Score Confidence Intervals
    results.ci_original = wilson_score_interval(
        data.original_successful, 
        data.evaluated
    )
    results.ci_improved = wilson_score_interval(
        data.improved_successful,
        data.evaluated
    )
    results.ci_reduction = newcombe_wilson_ci(
        data.original_success_rate, data.evaluated,
        data.improved_success_rate, data.evaluated
    )
    
    # 3. McNemar's Test
    # b = newly blocked (was successful, now failed)
    # c = newly successful (was failed, now successful)
    chi2, p_value = mcnemar_test(data.newly_blocked, data.newly_successful)
    results.mcnemar_chi2 = chi2
    results.mcnemar_p_value = p_value
    results.is_significant = p_value < 0.05
    
    # 4. Additional Metrics
    results.odds_ratio = calculate_odds_ratio(
        data.newly_blocked, 
        data.newly_successful
    )
    results.relative_risk = calculate_relative_risk(
        data.original_success_rate,
        data.improved_success_rate
    )
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualization(results_with: StatisticalResults, 
                        results_without: StatisticalResults,
                        output_path: str):
    """
    Create comprehensive multi-panel visualization.
    
    Panels:
    1. Attack Success Rates with CI (bar chart with error bars)
    2. Defense Improvement with CI (bar chart with error bars)
    3. Effect Size Comparison (Cohen's h with thresholds)
    4. McNemar's Test Results (contingency summary)
    5. Summary Statistics Table
    6. Interpretation Guide
    """
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3,
                          height_ratios=[1, 1, 0.8])
    
    # Color scheme
    color_with = '#3498db'
    color_without = '#2980b9'
    color_blocked = '#27ae60'
    color_reduction = '#16a085'
    
    scenarios = ['With AC', 'Without AC']
    x_pos = np.array([0, 1])
    
    # Extract data
    data_with = results_with.scenario
    data_without = results_without.scenario
    
    # =========================================================================
    # Panel 1: Original vs Improved Success Rates with CI
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    width = 0.35
    
    # Original rates
    orig_rates = [data_with.original_success_rate * 100, 
                  data_without.original_success_rate * 100]
    # Improved rates
    impr_rates = [data_with.improved_success_rate * 100,
                  data_without.improved_success_rate * 100]
    
    # CI for improved rates
    ci_with = results_with.ci_improved
    ci_without = results_without.ci_improved
    
    yerr_impr = [
        [
            (data_with.improved_success_rate - ci_with[0]) * 100,
            (data_without.improved_success_rate - ci_without[0]) * 100
        ],
        [
            (ci_with[1] - data_with.improved_success_rate) * 100,
            (ci_without[1] - data_without.improved_success_rate) * 100
        ]
    ]
    
    bars1 = ax1.bar(x_pos - width/2, orig_rates, width, label='Original', 
                    color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, impr_rates, width, label='After Defense',
                    color=color_with, alpha=0.8)
    ax1.errorbar(x_pos + width/2, impr_rates, yerr=yerr_impr, 
                 fmt='none', color='black', capsize=5, capthick=2)
    
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Attack Success Rate: Before vs After Defense\n(95% CI shown for After Defense)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'With AC\n(n={data_with.evaluated})', 
                         f'Without AC\n(n={data_without.evaluated})'])
    ax1.set_ylim([0, 120])
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels - positioned to avoid overlap
    for i, (orig, impr) in enumerate(zip(orig_rates, impr_rates)):
        ax1.text(i - width/2, orig + 3, f'{orig:.1f}%', ha='center', fontsize=10, fontweight='bold')
        ax1.text(i + width/2, impr + 12, f'{impr:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    # =========================================================================
    # Panel 2: Defense Improvement (Reduction) with CI
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    reductions = [data_with.success_rate_reduction * 100,
                  data_without.success_rate_reduction * 100]
    
    # CI for reduction
    ci_red_with = results_with.ci_reduction
    ci_red_without = results_without.ci_reduction
    
    yerr_red = [
        [
            (data_with.success_rate_reduction - ci_red_with[0]) * 100,
            (data_without.success_rate_reduction - ci_red_without[0]) * 100
        ],
        [
            (ci_red_with[1] - data_with.success_rate_reduction) * 100,
            (ci_red_without[1] - data_without.success_rate_reduction) * 100
        ]
    ]
    
    bars3 = ax2.bar(x_pos, reductions, 0.6, color=[color_blocked, color_reduction], alpha=0.8)
    ax2.errorbar(x_pos, reductions, yerr=yerr_red, fmt='none', color='black', capsize=5, capthick=2)
    
    ax2.set_ylabel('Success Rate Reduction (%)', fontsize=12)
    ax2.set_title('Defense Improvement\n(with 95% Confidence Interval)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(scenarios)
    ax2.set_ylim([0, max(reductions) * 1.5])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (red, blocked) in enumerate(zip(reductions, 
                                           [data_with.newly_blocked, data_without.newly_blocked])):
        ax2.text(i, red + max(reductions)*0.08, f'{red:.1f}%', 
                ha='center', fontsize=12, fontweight='bold')
        ax2.text(i, red + max(reductions)*0.18, f'({blocked} blocked)', 
                ha='center', fontsize=10)
    
    # =========================================================================
    # Panel 3: Cohen's h Effect Size
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    effect_sizes = [abs(results_with.cohens_h), abs(results_without.cohens_h)]
    
    # Color based on effect size magnitude
    colors_h = []
    for h in effect_sizes:
        if h >= 0.8:
            colors_h.append('#c0392b')  # Large - red
        elif h >= 0.5:
            colors_h.append('#f39c12')  # Medium - orange
        else:
            colors_h.append('#3498db')  # Small - blue
    
    bars4 = ax3.bar(x_pos, effect_sizes, 0.6, color=colors_h, alpha=0.8)
    
    # Threshold lines
    ax3.axhline(y=0.2, color='#27ae60', linestyle='--', linewidth=1.5, 
                label='Small (0.2)', alpha=0.7)
    ax3.axhline(y=0.5, color='#f39c12', linestyle='--', linewidth=1.5,
                label='Medium (0.5)', alpha=0.7)
    ax3.axhline(y=0.8, color='#c0392b', linestyle='--', linewidth=1.5,
                label='Large (0.8)', alpha=0.7)
    
    ax3.set_ylabel("Cohen's h (Effect Size)", fontsize=12)
    ax3.set_title("Effect Size: Cohen's h\n(Magnitude of Defense Impact)", fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(scenarios)
    ax3.set_ylim([0, max(effect_sizes) * 1.4])
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (h, interp) in enumerate(zip(effect_sizes, 
                                        [results_with.cohens_h_interpretation,
                                         results_without.cohens_h_interpretation])):
        ax3.text(i, h + max(effect_sizes)*0.05, f'h = {h:.3f}', 
                ha='center', fontsize=11, fontweight='bold')
        ax3.text(i, h + max(effect_sizes)*0.12, f'({interp})', 
                ha='center', fontsize=10, style='italic')
    
    # =========================================================================
    # Panel 4: McNemar's Test Visualization
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Create grouped bar for McNemar components
    mcnemar_data = {
        'Newly Blocked\n(Defense Success)': [data_with.newly_blocked, data_without.newly_blocked],
        'Newly Successful\n(Defense Failure)': [data_with.newly_successful, data_without.newly_successful]
    }
    
    x = np.arange(len(mcnemar_data))
    width = 0.35
    
    bars_with = ax4.bar(x - width/2, [mcnemar_data[k][0] for k in mcnemar_data], 
                        width, label='With AC', color=color_with, alpha=0.8)
    bars_without = ax4.bar(x + width/2, [mcnemar_data[k][1] for k in mcnemar_data],
                           width, label='Without AC', color=color_without, alpha=0.8)
    
    ax4.set_ylabel('Number of Attacks', fontsize=12)
    ax4.set_title("McNemar's Test Components\n(Paired Changes)", fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(list(mcnemar_data.keys()))
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars_with, bars_without]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # =========================================================================
    # Panel 5: Statistical Summary Table
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    
    table_data = [
        ['Metric', 'With AC', 'Without AC'],
        ['Sample Size (n)', f'{data_with.evaluated}', f'{data_without.evaluated}'],
        ['Original Success Rate', f'{data_with.original_success_rate*100:.1f}%', 
                                  f'{data_without.original_success_rate*100:.1f}%'],
        ['After Defense Rate', f'{data_with.improved_success_rate*100:.1f}%',
                               f'{data_without.improved_success_rate*100:.1f}%'],
        ['95% CI (After)', f'[{results_with.ci_improved[0]*100:.1f}%, {results_with.ci_improved[1]*100:.1f}%]',
                           f'[{results_without.ci_improved[0]*100:.1f}%, {results_without.ci_improved[1]*100:.1f}%]'],
        ['Reduction', f'{data_with.success_rate_reduction*100:.1f}%',
                      f'{data_without.success_rate_reduction*100:.1f}%'],
        ['95% CI (Reduction)', f'[{results_with.ci_reduction[0]*100:.1f}%, {results_with.ci_reduction[1]*100:.1f}%]',
                               f'[{results_without.ci_reduction[0]*100:.1f}%, {results_without.ci_reduction[1]*100:.1f}%]'],
        ['Newly Blocked', f'{data_with.newly_blocked}', f'{data_without.newly_blocked}'],
        ['Newly Successful', f'{data_with.newly_successful}', f'{data_without.newly_successful}'],
    ]
    
    table = ax5.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.45, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    
    # Style header
    for j in range(3):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    ax5.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=20)
    
    # =========================================================================
    # Panel 6: Statistical Test Results
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stat_table = [
        ['Statistical Test', 'With AC', 'Without AC'],
        ["Cohen's h", f'{abs(results_with.cohens_h):.3f}', f'{abs(results_without.cohens_h):.3f}'],
        ['Effect Interpretation', results_with.cohens_h_interpretation, 
                                  results_without.cohens_h_interpretation],
        ["McNemar's Chi-square", f'{results_with.mcnemar_chi2:.3f}', 
                                 f'{results_without.mcnemar_chi2:.3f}'],
        ['p-value', f'{results_with.mcnemar_p_value:.4f}', 
                    f'{results_without.mcnemar_p_value:.4f}'],
        ['Significant (p<0.05)', 'Yes' if results_with.is_significant else 'No',
                                 'Yes' if results_without.is_significant else 'No'],
        ['Odds Ratio (b/c)', f'{results_with.odds_ratio:.2f}' if results_with.odds_ratio != float('inf') else 'Inf',
                             f'{results_without.odds_ratio:.2f}' if results_without.odds_ratio != float('inf') else 'Inf'],
    ]
    
    table2 = ax6.table(cellText=stat_table, loc='center', cellLoc='center',
                       colWidths=[0.45, 0.35, 0.35])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.0, 1.6)
    
    # Style header
    for j in range(3):
        table2[(0, j)].set_facecolor('#27ae60')
        table2[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Highlight significance
    for i in range(1, len(stat_table)):
        for j in range(3):
            if 'Yes' in str(stat_table[i][j]):
                table2[(i, j)].set_facecolor('#d5f4e6')
            elif 'No' in str(stat_table[i][j]) and i == 5:
                table2[(i, j)].set_facecolor('#fadbd8')
    
    ax6.set_title('Statistical Test Results', fontsize=13, fontweight='bold', pad=20)
    
    # =========================================================================
    # Panel 7: Key Findings (Bottom spanning panel)
    # =========================================================================
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    findings_text = f"""
KEY FINDINGS
{'='*100}

1. EFFECT SIZE COMPARISON
   - With AC: Cohen's h = {abs(results_with.cohens_h):.3f} ({results_with.cohens_h_interpretation} effect)
   - Without AC: Cohen's h = {abs(results_without.cohens_h):.3f} ({results_without.cohens_h_interpretation} effect)
   - Winner: {'Without AC' if abs(results_without.cohens_h) > abs(results_with.cohens_h) else 'With AC'} shows larger effect size

2. STATISTICAL SIGNIFICANCE (McNemar's Test)
   - With AC: p = {results_with.mcnemar_p_value:.4f} ({'Significant' if results_with.is_significant else 'Not Significant'} at alpha=0.05)
   - Without AC: p = {results_without.mcnemar_p_value:.4f} ({'Significant' if results_without.is_significant else 'Not Significant'} at alpha=0.05)

3. PRACTICAL IMPLICATIONS
   - Systems without explicit access codes show {abs(data_without.success_rate_reduction - data_with.success_rate_reduction)*100:.1f} percentage points BETTER defense improvement
   - Explicit vulnerability indicators in system prompts may make effective defense more challenging
   - Defense blocked {data_with.newly_blocked + data_without.newly_blocked} attacks total across both scenarios

4. CONFIDENCE IN RESULTS
   - 95% CI for With AC reduction: [{results_with.ci_reduction[0]*100:.1f}%, {results_with.ci_reduction[1]*100:.1f}%]
   - 95% CI for Without AC reduction: [{results_without.ci_reduction[0]*100:.1f}%, {results_without.ci_reduction[1]*100:.1f}%]
"""
    
    ax7.text(0.02, 0.95, findings_text, transform=ax7.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8, edgecolor='#dee2e6'))
    
    # Main title
    fig.suptitle('LLM Prompt Injection Defense: Statistical Analysis\nComparing Systems With vs Without Access Codes (AC)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"\nVisualization saved: {output_path}")
    plt.close()


# =============================================================================
# TEXT REPORT GENERATION
# =============================================================================

def generate_detailed_report(results_with: StatisticalResults,
                            results_without: StatisticalResults,
                            output_path: str):
    """Generate comprehensive text report"""
    
    data_with = results_with.scenario
    data_without = results_without.scenario
    
    # Pre-calculate formatted strings
    odds_ratio_with = f"{results_with.odds_ratio:.2f}" if results_with.odds_ratio != float('inf') else "Infinity (c=0)"
    odds_ratio_without = f"{results_without.odds_ratio:.2f}" if results_without.odds_ratio != float('inf') else "Infinity (c=0)"
    
    report = f"""
================================================================================
LLM PROMPT INJECTION DEFENSE: STATISTICAL ANALYSIS REPORT
================================================================================

Research Question: Does including access codes in system prompts affect 
the effectiveness of prompt injection defense?

Data Source: Duke University Cybersecurity 590 Research Project
Analysis Date: November 2024

================================================================================
SECTION 1: DATA SUMMARY
================================================================================

SCENARIO 1: WITH ACCESS CODE IN SYSTEM PROMPT
----------------------------------------------
Total attacks in dataset:       {data_with.total_dataset}
Attacks with improved prompts:  {data_with.evaluated} (evaluated)
Attacks without improvements:   {data_with.skipped} (skipped)

Original Results (Among Evaluated):
  - Successful attacks:         {data_with.original_successful} ({data_with.original_success_rate*100:.1f}%)
  - Failed attacks:             {data_with.original_failed} ({(1-data_with.original_success_rate)*100:.1f}%)

After Defense:
  - Successful attacks:         {data_with.improved_successful} ({data_with.improved_success_rate*100:.1f}%)
  - Failed attacks:             {data_with.improved_failed} ({(1-data_with.improved_success_rate)*100:.1f}%)

Changes:
  - Newly blocked by defense:   {data_with.newly_blocked}
  - Newly successful (worse):   {data_with.newly_successful}


SCENARIO 2: WITHOUT ACCESS CODE IN SYSTEM PROMPT
-------------------------------------------------
Total attacks in dataset:       {data_without.total_dataset}
Attacks with improved prompts:  {data_without.evaluated} (evaluated)
Attacks without improvements:   {data_without.skipped} (skipped)

Original Results (Among Evaluated):
  - Successful attacks:         {data_without.original_successful} ({data_without.original_success_rate*100:.1f}%)
  - Failed attacks:             {data_without.original_failed} ({(1-data_without.original_success_rate)*100:.1f}%)

After Defense:
  - Successful attacks:         {data_without.improved_successful} ({data_without.improved_success_rate*100:.1f}%)
  - Failed attacks:             {data_without.improved_failed} ({(1-data_without.improved_success_rate)*100:.1f}%)

Changes:
  - Newly blocked by defense:   {data_without.newly_blocked}
  - Newly successful (worse):   {data_without.newly_successful}


================================================================================
SECTION 2: EFFECT SIZE ANALYSIS (COHEN'S H)
================================================================================

Cohen's h measures the magnitude of difference between two proportions.
Formula: h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

INTERPRETATION SCALE:
  |h| < 0.2   Small effect
  |h| = 0.5   Medium effect  
  |h| > 0.8   Large effect

RESULTS:

With Access Code:
  Original rate:    {data_with.original_success_rate*100:.1f}%
  Improved rate:    {data_with.improved_success_rate*100:.1f}%
  Cohen's h:        {results_with.cohens_h:.4f}
  |h|:              {abs(results_with.cohens_h):.4f}
  Interpretation:   {results_with.cohens_h_interpretation} effect

Without Access Code:
  Original rate:    {data_without.original_success_rate*100:.1f}%
  Improved rate:    {data_without.improved_success_rate*100:.1f}%
  Cohen's h:        {results_without.cohens_h:.4f}
  |h|:              {abs(results_without.cohens_h):.4f}
  Interpretation:   {results_without.cohens_h_interpretation} effect

COMPARISON:
  Difference in |h|: {abs(abs(results_without.cohens_h) - abs(results_with.cohens_h)):.4f}
  Larger effect:     {'Without Access Code' if abs(results_without.cohens_h) > abs(results_with.cohens_h) else 'With Access Code'}


================================================================================
SECTION 3: CONFIDENCE INTERVALS (WILSON SCORE METHOD)
================================================================================

The Wilson score interval provides more accurate coverage than the normal 
approximation, especially for small samples or extreme proportions.

95% CONFIDENCE INTERVALS:

With Access Code (n={data_with.evaluated}):
  Original Success Rate:
    Point estimate:   {data_with.original_success_rate*100:.1f}%
    95% CI:           [{results_with.ci_original[0]*100:.1f}%, {results_with.ci_original[1]*100:.1f}%]
    
  Improved Success Rate:
    Point estimate:   {data_with.improved_success_rate*100:.1f}%
    95% CI:           [{results_with.ci_improved[0]*100:.1f}%, {results_with.ci_improved[1]*100:.1f}%]
    
  Success Rate Reduction:
    Point estimate:   {data_with.success_rate_reduction*100:.1f}%
    95% CI:           [{results_with.ci_reduction[0]*100:.1f}%, {results_with.ci_reduction[1]*100:.1f}%]

Without Access Code (n={data_without.evaluated}):
  Original Success Rate:
    Point estimate:   {data_without.original_success_rate*100:.1f}%
    95% CI:           [{results_without.ci_original[0]*100:.1f}%, {results_without.ci_original[1]*100:.1f}%]
    
  Improved Success Rate:
    Point estimate:   {data_without.improved_success_rate*100:.1f}%
    95% CI:           [{results_without.ci_improved[0]*100:.1f}%, {results_without.ci_improved[1]*100:.1f}%]
    
  Success Rate Reduction:
    Point estimate:   {data_without.success_rate_reduction*100:.1f}%
    95% CI:           [{results_without.ci_reduction[0]*100:.1f}%, {results_without.ci_reduction[1]*100:.1f}%]

NOTE: Wider intervals for "Without Access Code" due to smaller sample size (n=20 vs n=38).


================================================================================
SECTION 4: MCNEMAR'S TEST FOR PAIRED DATA
================================================================================

McNemar's test assesses whether the changes in attack outcomes (blocked vs 
newly successful) are statistically significant.

CONTINGENCY TABLE STRUCTURE:
                        After Defense
                    Success     Failure
Before:  Success      a           b (blocked)
         Failure      c (new)     d

Test statistic: Chi-square = (|b - c| - 1)^2 / (b + c)  [with continuity correction]

RESULTS:

With Access Code:
  Newly blocked (b):        {data_with.newly_blocked}
  Newly successful (c):     {data_with.newly_successful}
  Chi-square statistic:     {results_with.mcnemar_chi2:.4f}
  p-value:                  {results_with.mcnemar_p_value:.4f}
  Significant (alpha=0.05): {'YES' if results_with.is_significant else 'NO'}
  Odds Ratio (b/c):         {odds_ratio_with}

Without Access Code:
  Newly blocked (b):        {data_without.newly_blocked}
  Newly successful (c):     {data_without.newly_successful}
  Chi-square statistic:     {results_without.mcnemar_chi2:.4f}
  p-value:                  {results_without.mcnemar_p_value:.4f}
  Significant (alpha=0.05): {'YES' if results_without.is_significant else 'NO'}
  Odds Ratio (b/c):         {odds_ratio_without}

INTERPRETATION:
  - With Access Code: The defense improvement is {'statistically significant' if results_with.is_significant else 'NOT statistically significant'}.
  - Without Access Code: The defense improvement is {'statistically significant' if results_without.is_significant else 'NOT statistically significant'}.


================================================================================
SECTION 5: COMPARATIVE ANALYSIS
================================================================================

DEFENSE EFFECTIVENESS COMPARISON:

                                With Code       Without Code    Difference
--------------------------------------------------------------------------------
Success Rate Reduction          {data_with.success_rate_reduction*100:.1f}%            {data_without.success_rate_reduction*100:.1f}%            {(data_without.success_rate_reduction - data_with.success_rate_reduction)*100:+.1f}%
Effect Size (|h|)               {abs(results_with.cohens_h):.3f}            {abs(results_without.cohens_h):.3f}            {abs(results_without.cohens_h) - abs(results_with.cohens_h):+.3f}
p-value                         {results_with.mcnemar_p_value:.4f}          {results_without.mcnemar_p_value:.4f}          --
Attacks Blocked                 {data_with.newly_blocked}                 {data_without.newly_blocked}                 {data_without.newly_blocked - data_with.newly_blocked:+d}
Attacks Made Worse              {data_with.newly_successful}                 {data_without.newly_successful}                 {data_without.newly_successful - data_with.newly_successful:+d}

WINNER BY METRIC:
  - Higher Reduction:           {'Without Access Code' if data_without.success_rate_reduction > data_with.success_rate_reduction else 'With Access Code'}
  - Larger Effect Size:         {'Without Access Code' if abs(results_without.cohens_h) > abs(results_with.cohens_h) else 'With Access Code'}
  - Statistical Significance:   {'Without Access Code' if results_without.is_significant and not results_with.is_significant else 'Both' if results_with.is_significant and results_without.is_significant else 'With Access Code' if results_with.is_significant else 'Neither'}


================================================================================
SECTION 6: KEY FINDINGS AND CONCLUSIONS
================================================================================

FINDING 1: EFFECT SIZE
Both scenarios show {'substantial' if min(abs(results_with.cohens_h), abs(results_without.cohens_h)) >= 0.5 else 'moderate'} effect sizes,
indicating that the defense prompts have {'meaningful' if min(abs(results_with.cohens_h), abs(results_without.cohens_h)) >= 0.5 else 'some'} practical impact.
- With Access Code: |h| = {abs(results_with.cohens_h):.3f} ({results_with.cohens_h_interpretation})
- Without Access Code: |h| = {abs(results_without.cohens_h):.3f} ({results_without.cohens_h_interpretation})

FINDING 2: STATISTICAL SIGNIFICANCE
{'Only the Without Access Code scenario shows statistically significant improvement (p < 0.05).' if results_without.is_significant and not results_with.is_significant else 'Both scenarios show statistically significant improvement (p < 0.05).' if results_with.is_significant and results_without.is_significant else 'Only the With Access Code scenario shows statistically significant improvement (p < 0.05).' if results_with.is_significant else 'Neither scenario shows statistically significant improvement at alpha = 0.05.'}

FINDING 3: PRACTICAL IMPLICATIONS
Systems without explicit access codes in system prompts show:
- {abs(data_without.success_rate_reduction - data_with.success_rate_reduction)*100:.1f} percentage points better defense improvement
- {abs(abs(results_without.cohens_h) - abs(results_with.cohens_h)):.3f} larger effect size

This suggests that explicit vulnerability indicators (like access codes) in 
system prompts may make it harder to develop effective defenses, possibly 
because attackers have a clear target to extract.

FINDING 4: CONFIDENCE IN RESULTS
The 95% confidence intervals provide uncertainty estimates:
- Results are robust within the stated ranges
- Smaller sample size for "Without Access Code" (n={data_without.evaluated}) results in wider intervals
- Despite wider intervals, the difference remains meaningful


================================================================================
SECTION 7: RECOMMENDATIONS FOR REPORTING
================================================================================

ACADEMIC CITATION FORMAT:

"Defense prompts reduced attack success rates by {data_with.success_rate_reduction*100:.1f}% 
(95% CI: [{results_with.ci_reduction[0]*100:.1f}%, {results_with.ci_reduction[1]*100:.1f}%]) for systems with access codes 
(Cohen's h = {abs(results_with.cohens_h):.2f}, McNemar's p = {results_with.mcnemar_p_value:.3f}) and by 
{data_without.success_rate_reduction*100:.1f}% (95% CI: [{results_without.ci_reduction[0]*100:.1f}%, {results_without.ci_reduction[1]*100:.1f}%]) 
for systems without access codes (Cohen's h = {abs(results_without.cohens_h):.2f}, 
McNemar's p = {results_without.mcnemar_p_value:.3f}). Both effect sizes are classified as 
'{results_with.cohens_h_interpretation}' and '{results_without.cohens_h_interpretation}' respectively 
according to Cohen's conventions."


================================================================================
SECTION 8: METHODOLOGY NOTES
================================================================================

STATISTICAL METHODS USED:

1. Cohen's h Effect Size
   - Measures magnitude of difference between proportions
   - Independent of sample size
   - Allows comparison across studies

2. Wilson Score Confidence Intervals
   - More accurate than normal approximation
   - Performs well for small samples
   - Handles proportions near 0 or 1

3. McNemar's Test
   - Appropriate for paired nominal data
   - Tests marginal homogeneity
   - Uses continuity correction for small samples

4. Newcombe-Wilson Method
   - For CI of difference between proportions
   - Combines Wilson intervals appropriately

LIMITATIONS:

1. Sample sizes differ between scenarios (n={data_with.evaluated} vs n={data_without.evaluated})
2. Only evaluated attacks with improved prompts
3. Heuristic-based defense assessment
4. Single iteration of defense improvement


================================================================================
END OF REPORT
================================================================================

Generated by: LLM Security Research Statistical Analysis Script
Project: Duke University Cybersecurity 590
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Detailed report saved: {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    print("="*70)
    print("LLM PROMPT INJECTION DEFENSE: STATISTICAL ANALYSIS")
    print("="*70)
    print("\nUsing real project data from:")
    print("D:\\LLM_judge\\results\\defense_evaluation_filtered\\")
    print()
    
    # Create scenario data objects from real data
    data_with = ScenarioData(
        name="With Access Code",
        **REAL_DATA['with_access_code']
    )
    
    data_without = ScenarioData(
        name="Without Access Code",
        **REAL_DATA['without_access_code']
    )
    
    # Perform statistical analysis
    print("Performing statistical analysis...")
    print("-"*70)
    
    results_with = perform_statistical_analysis(data_with)
    results_without = perform_statistical_analysis(data_without)
    
    # Print summary to console
    print(f"\n{'SCENARIO':<25} {'With Access Code':<20} {'Without Access Code':<20}")
    print("-"*70)
    print(f"{'Sample Size (n)':<25} {data_with.evaluated:<20} {data_without.evaluated:<20}")
    print(f"{'Original Success Rate':<25} {data_with.original_success_rate*100:.1f}%{'':<15} {data_without.original_success_rate*100:.1f}%")
    print(f"{'After Defense Rate':<25} {data_with.improved_success_rate*100:.1f}%{'':<15} {data_without.improved_success_rate*100:.1f}%")
    print(f"{'Success Rate Reduction':<25} {data_with.success_rate_reduction*100:.1f}%{'':<15} {data_without.success_rate_reduction*100:.1f}%")
    print(f"{'95% CI (Reduction)':<25} [{results_with.ci_reduction[0]*100:.1f}%, {results_with.ci_reduction[1]*100:.1f}%]{'':<5} [{results_without.ci_reduction[0]*100:.1f}%, {results_without.ci_reduction[1]*100:.1f}%]")
    print(f"{'Cohens h':<25} {abs(results_with.cohens_h):.3f}{'':<16} {abs(results_without.cohens_h):.3f}")
    print(f"{'Effect Interpretation':<25} {results_with.cohens_h_interpretation:<20} {results_without.cohens_h_interpretation:<20}")
    print(f"{'McNemars p-value':<25} {results_with.mcnemar_p_value:.4f}{'':<14} {results_without.mcnemar_p_value:.4f}")
    sig_with = "Yes" if results_with.is_significant else "No"
    sig_without = "Yes" if results_without.is_significant else "No"
    print(f"{'Significant (p<0.05)':<25} {sig_with:<20} {sig_without:<20}")
    print("-"*70)
    
    # Create output directory
    output_dir = Path('D:/LLM_judge/results/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualization
    viz_path = output_dir / 'defense_statistical_analysis.png'
    create_visualization(results_with, results_without, str(viz_path))
    
    # Generate detailed report
    report_path = output_dir / 'defense_statistical_analysis_report.txt'
    generate_detailed_report(results_with, results_without, str(report_path))
    
    # Summary
    print("\n" + "="*70)
    print("OUTPUT FILES GENERATED")
    print("="*70)
    print(f"1. Visualization:  {viz_path}")
    print(f"2. Detailed Report: {report_path}")
    print("="*70)
    
    print("\nKEY FINDINGS:")
    print(f"- Without Access Code shows {abs(data_without.success_rate_reduction - data_with.success_rate_reduction)*100:.1f}% better defense improvement")
    print(f"- Effect sizes: {results_with.cohens_h_interpretation} (with) vs {results_without.cohens_h_interpretation} (without)")
    print(f"- Statistical significance: {'Both significant' if results_with.is_significant and results_without.is_significant else 'Only Without Code significant' if results_without.is_significant else 'Only With Code significant' if results_with.is_significant else 'Neither significant'}")
    

if __name__ == "__main__":
    main()