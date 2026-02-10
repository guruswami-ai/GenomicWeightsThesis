#!/usr/bin/env python3
"""
Analysis script for Genomic Weight Thesis multi-seed experiments.

Reads JSON results from results/ directory and produces:
- Summary statistics table (for paper)
- Fitness convergence plots
- Statistical significance tests

Usage: python3 analyze_results.py [results_dir]
"""
import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np


def load_results(results_dir: Path) -> Dict[str, List[dict]]:
    """Load all JSON results, grouped by strategy."""
    results_by_strategy = {'flat': [], 'hierarchical': [], 'topological': []}

    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            strategy = data.get('strategy', 'unknown')
            if strategy in results_by_strategy:
                # Add all successful runs
                for run in data.get('runs', []):
                    if run.get('success', False):
                        run['experiment_id'] = data.get('experiment_id', json_file.stem)
                        results_by_strategy[strategy].append(run)

        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return results_by_strategy


def compute_statistics(runs: List[dict]) -> dict:
    """Compute summary statistics for a set of runs."""
    if not runs:
        return {'n': 0}

    fitnesses = [r['final_fitness'] for r in runs]
    times = [r['total_time'] for r in runs]
    steps_per_sec = [r['avg_steps_per_sec'] for r in runs]

    return {
        'n': len(runs),
        'mean_fitness': np.mean(fitnesses),
        'std_fitness': np.std(fitnesses),
        'max_fitness': np.max(fitnesses),
        'min_fitness': np.min(fitnesses),
        'median_fitness': np.median(fitnesses),
        'mean_time': np.mean(times),
        'mean_steps_per_sec': np.mean(steps_per_sec),
        'fitnesses': fitnesses  # For statistical tests
    }


def welch_ttest(a: List[float], b: List[float]) -> tuple:
    """Welch's t-test for unequal variances."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return None, None

    mean1, mean2 = np.mean(a), np.mean(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)

    # Welch's t-statistic
    se = np.sqrt(var1/n1 + var2/n2)
    if se == 0:
        return None, None

    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var1/n1 + var2/n2)**2
    denom = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
    df = num / denom if denom > 0 else 1

    # Two-tailed p-value approximation (using normal for large df)
    from math import erfc, sqrt
    p_value = erfc(abs(t_stat) / sqrt(2))

    return t_stat, p_value


def cohens_d(a: List[float], b: List[float]) -> float:
    """Cohen's d effect size."""
    if len(a) < 2 or len(b) < 2:
        return None

    mean1, mean2 = np.mean(a), np.mean(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    n1, n2 = len(a), len(b)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    if pooled_std == 0:
        return None

    return (mean1 - mean2) / pooled_std


def print_summary_table(stats_by_strategy: Dict[str, dict]):
    """Print markdown summary table for paper."""
    print("\n## Summary Statistics\n")
    print("| Strategy | N | Mean Fitness | Std | Max | Min | Median |")
    print("|----------|---|--------------|-----|-----|-----|--------|")

    for strategy in ['flat', 'hierarchical', 'topological']:
        s = stats_by_strategy.get(strategy, {'n': 0})
        if s['n'] > 0:
            print(f"| {strategy.capitalize()} | {s['n']} | "
                  f"{s['mean_fitness']:.2f} | {s['std_fitness']:.2f} | "
                  f"{s['max_fitness']:.2f} | {s['min_fitness']:.2f} | "
                  f"{s['median_fitness']:.2f} |")
        else:
            print(f"| {strategy.capitalize()} | 0 | - | - | - | - | - |")


def print_pairwise_tests(stats_by_strategy: Dict[str, dict]):
    """Print pairwise statistical comparisons."""
    print("\n## Pairwise Comparisons (Welch's t-test)\n")
    print("| Comparison | t-statistic | p-value | Cohen's d | Interpretation |")
    print("|------------|-------------|---------|-----------|----------------|")

    comparisons = [
        ('hierarchical', 'flat'),
        ('topological', 'flat'),
        ('hierarchical', 'topological')
    ]

    for strat1, strat2 in comparisons:
        s1 = stats_by_strategy.get(strat1, {})
        s2 = stats_by_strategy.get(strat2, {})

        if s1.get('n', 0) < 2 or s2.get('n', 0) < 2:
            print(f"| {strat1} vs {strat2} | - | - | - | Insufficient data |")
            continue

        t_stat, p_value = welch_ttest(s1['fitnesses'], s2['fitnesses'])
        d = cohens_d(s1['fitnesses'], s2['fitnesses'])

        if t_stat is None:
            print(f"| {strat1} vs {strat2} | - | - | - | Could not compute |")
            continue

        # Interpretation
        if p_value < 0.01:
            sig = "**Significant (p<0.01)**"
        elif p_value < 0.05:
            sig = "*Significant (p<0.05)*"
        else:
            sig = "Not significant"

        if d is not None:
            if abs(d) > 0.8:
                effect = "Large"
            elif abs(d) > 0.5:
                effect = "Medium"
            elif abs(d) > 0.2:
                effect = "Small"
            else:
                effect = "Negligible"
            d_str = f"{d:.3f} ({effect})"
        else:
            d_str = "-"

        print(f"| {strat1} vs {strat2} | {t_stat:.3f} | {p_value:.4f} | {d_str} | {sig} |")


def print_convergence_data(results_by_strategy: Dict[str, List[dict]]):
    """Print convergence curve data for plotting."""
    print("\n## Convergence Data (Mean Max Fitness per Generation)\n")

    for strategy in ['flat', 'hierarchical', 'topological']:
        runs = results_by_strategy.get(strategy, [])
        if not runs:
            continue

        # Get all fitness curves
        curves = [r['fitness_curve'] for r in runs if 'fitness_curve' in r]
        if not curves:
            continue

        # Compute mean across runs for each generation
        min_len = min(len(c) for c in curves)
        mean_curve = []
        std_curve = []

        for gen in range(min_len):
            vals = [c[gen] for c in curves]
            mean_curve.append(np.mean(vals))
            std_curve.append(np.std(vals))

        print(f"### {strategy.capitalize()}")
        print(f"Generations: {min_len}, Runs: {len(curves)}")
        print(f"Final mean: {mean_curve[-1]:.2f} +/- {std_curve[-1]:.2f}")
        print()


def save_convergence_csv(results_by_strategy: Dict[str, List[dict]], output_dir: Path):
    """Save convergence data as CSV for external plotting."""
    output_dir.mkdir(exist_ok=True)

    for strategy in ['flat', 'hierarchical', 'topological']:
        runs = results_by_strategy.get(strategy, [])
        if not runs:
            continue

        curves = [r['fitness_curve'] for r in runs if 'fitness_curve' in r]
        if not curves:
            continue

        min_len = min(len(c) for c in curves)

        csv_path = output_dir / f"convergence_{strategy}.csv"
        with open(csv_path, 'w') as f:
            # Header
            f.write("generation," + ",".join(f"seed_{i}" for i in range(len(curves))) + ",mean,std\n")

            for gen in range(min_len):
                vals = [c[gen] for c in curves]
                row = [str(gen)] + [f"{v:.2f}" for v in vals]
                row.append(f"{np.mean(vals):.2f}")
                row.append(f"{np.std(vals):.2f}")
                f.write(",".join(row) + "\n")

        print(f"Saved: {csv_path}")


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run experiments first with run_multiseed.py")
        sys.exit(1)

    print(f"Loading results from: {results_dir}")
    results_by_strategy = load_results(results_dir)

    # Count runs
    total_runs = sum(len(runs) for runs in results_by_strategy.values())
    print(f"Loaded {total_runs} successful runs")
    for strategy, runs in results_by_strategy.items():
        print(f"  {strategy}: {len(runs)} runs")

    if total_runs == 0:
        print("No results found. Experiments may still be running.")
        sys.exit(0)

    # Compute statistics
    stats_by_strategy = {
        strategy: compute_statistics(runs)
        for strategy, runs in results_by_strategy.items()
    }

    # Print results for paper
    print("\n" + "="*70)
    print("RESULTS FOR PAPER")
    print("="*70)

    print_summary_table(stats_by_strategy)
    print_pairwise_tests(stats_by_strategy)
    print_convergence_data(results_by_strategy)

    # Save CSV for plotting
    save_convergence_csv(results_by_strategy, results_dir / "csv")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
