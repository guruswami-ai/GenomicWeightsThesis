#!/usr/bin/env python3
"""
Validation utilities for experiment integrity.
Detects silent failures, NaN propagation, and invalid states.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, List
import json
import time

class ExperimentValidator:
    """Tracks experiment health and detects failure states"""
    
    def __init__(self, strategy: str, seed: int):
        self.strategy = strategy
        self.seed = seed
        self.issues = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_failed = 0
        
    def log_issue(self, severity: str, message: str, data: dict = None):
        """Log an issue with severity: ERROR, WARNING, INFO"""
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "severity": severity,
            "message": message,
            "data": data or {}
        }
        if severity == "ERROR":
            self.issues.append(entry)
            self.checks_failed += 1
            print(f"❌ ERROR: {message}")
        elif severity == "WARNING":
            self.warnings.append(entry)
            print(f"⚠️ WARNING: {message}")
        else:
            self.checks_passed += 1
            
    def validate_phenotype_output(self, action: jnp.ndarray, generation: int) -> bool:
        """Check phenotype network produces valid actions"""
        issues_found = False
        
        # Check for NaN
        if jnp.any(jnp.isnan(action)):
            self.log_issue("ERROR", f"Gen {generation}: NaN in action output", 
                          {"action": str(action)})
            issues_found = True
            
        # Check for Inf
        if jnp.any(jnp.isinf(action)):
            self.log_issue("ERROR", f"Gen {generation}: Inf in action output",
                          {"action": str(action)})
            issues_found = True
            
        # Check action bounds (should be [-1, 1] after tanh)
        if jnp.any(jnp.abs(action) > 1.0 + 1e-6):
            self.log_issue("ERROR", f"Gen {generation}: Action out of bounds",
                          {"min": float(jnp.min(action)), "max": float(jnp.max(action))})
            issues_found = True
            
        # Check for all-zeros (degenerate solution)
        if jnp.allclose(action, 0.0, atol=1e-6):
            self.log_issue("WARNING", f"Gen {generation}: Action is all zeros (degenerate)")
            
        if not issues_found:
            self.checks_passed += 1
            
        return not issues_found
    
    def validate_fitness(self, fitness: float, generation: int, 
                        historical_fitness: List[float]) -> bool:
        """Check fitness values are reasonable"""
        issues_found = False
        
        # Check for NaN
        if np.isnan(fitness):
            self.log_issue("ERROR", f"Gen {generation}: NaN fitness")
            return False
            
        # Check for Inf
        if np.isinf(fitness):
            self.log_issue("ERROR", f"Gen {generation}: Inf fitness")
            return False
            
        # Check for extreme values (Brax Ant typically -100 to 1000)
        if fitness < -1000:
            self.log_issue("WARNING", f"Gen {generation}: Very low fitness {fitness:.2f}")
        if fitness > 5000:
            self.log_issue("WARNING", f"Gen {generation}: Suspiciously high fitness {fitness:.2f}")
            
        # Check for sudden collapse (fitness drops to near-zero)
        if len(historical_fitness) > 5:
            recent_avg = np.mean(historical_fitness[-5:])
            if recent_avg > 100 and fitness < 10:
                self.log_issue("WARNING", f"Gen {generation}: Fitness collapsed {recent_avg:.1f} → {fitness:.1f}")
                
        # Check for stagnation (no improvement in 100 generations)
        if len(historical_fitness) > 100:
            old_max = max(historical_fitness[-100:-50])
            new_max = max(historical_fitness[-50:])
            if new_max <= old_max * 1.01:  # Less than 1% improvement
                self.log_issue("WARNING", f"Gen {generation}: Fitness stagnant for 100 gens")
                
        self.checks_passed += 1
        return True
    
    def validate_genotype_weights(self, params_flat: np.ndarray, generation: int) -> bool:
        """Check genotype parameters are healthy"""
        issues_found = False
        
        # Check for NaN
        if np.any(np.isnan(params_flat)):
            self.log_issue("ERROR", f"Gen {generation}: NaN in genotype params")
            return False
            
        # Check for weight explosion
        max_weight = np.max(np.abs(params_flat))
        if max_weight > 100:
            self.log_issue("WARNING", f"Gen {generation}: Large weights detected (max={max_weight:.2f})")
            
        # Check for weight collapse
        std_weight = np.std(params_flat)
        if std_weight < 1e-6:
            self.log_issue("WARNING", f"Gen {generation}: Weight diversity collapsed (std={std_weight:.2e})")
            
        # Log weight statistics
        stats = {
            "mean": float(np.mean(params_flat)),
            "std": float(std_weight),
            "min": float(np.min(params_flat)),
            "max": float(np.max(params_flat)),
        }
        
        self.checks_passed += 1
        return True, stats
    
    def validate_phenotype_data(self, phenotype_data: Dict, generation: int) -> bool:
        """Check phenotype data structure is valid"""
        
        strategy = phenotype_data.get('strategy')
        if strategy != self.strategy:
            self.log_issue("ERROR", f"Gen {generation}: Strategy mismatch {strategy} != {self.strategy}")
            return False
            
        if strategy == 'flat':
            weights = phenotype_data.get('weights')
            if weights is None:
                self.log_issue("ERROR", f"Gen {generation}: Missing 'weights' in flat phenotype")
                return False
            if jnp.any(jnp.isnan(weights)):
                self.log_issue("ERROR", f"Gen {generation}: NaN in flat weights")
                return False
                
        elif strategy == 'hierarchical':
            blocks = phenotype_data.get('blocks')
            proj = phenotype_data.get('projection')
            if blocks is None or proj is None:
                self.log_issue("ERROR", f"Gen {generation}: Missing blocks/projection in hierarchical")
                return False
            for i, block in enumerate(blocks):
                if jnp.any(jnp.isnan(block)):
                    self.log_issue("ERROR", f"Gen {generation}: NaN in block {i}")
                    return False
                    
        elif strategy == 'topological':
            adj = phenotype_data.get('adjacency')
            proj = phenotype_data.get('projection')
            if adj is None or proj is None:
                self.log_issue("ERROR", f"Gen {generation}: Missing adjacency/projection in topological")
                return False
            if jnp.any(jnp.isnan(adj)):
                self.log_issue("ERROR", f"Gen {generation}: NaN in adjacency matrix")
                return False
            # Check adjacency is valid probability matrix
            row_sums = jnp.sum(adj, axis=-1)
            if not jnp.allclose(row_sums, 1.0, atol=0.1):
                self.log_issue("WARNING", f"Gen {generation}: Adjacency rows don't sum to 1")
                
        self.checks_passed += 1
        return True
    
    def get_summary(self) -> Dict:
        """Return validation summary"""
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "errors": len(self.issues),
            "warnings": len(self.warnings),
            "issues": self.issues,
            "warnings_list": self.warnings,
            "status": "FAILED" if self.issues else ("WARNING" if self.warnings else "PASSED")
        }
    
    def save_report(self, output_path: str):
        """Save validation report to file"""
        summary = self.get_summary()
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n📋 Validation report saved to: {output_path}")
        print(f"   Status: {summary['status']}")
        print(f"   Checks: {summary['checks_passed']} passed, {summary['checks_failed']} failed")
        if summary['warnings']:
            print(f"   Warnings: {len(summary['warnings'])}")


def quick_sanity_check(phenotype_net, obs: jnp.ndarray) -> Tuple[bool, str]:
    """Quick sanity check of phenotype network"""
    try:
        action = phenotype_net(obs)
        
        if jnp.any(jnp.isnan(action)):
            return False, "NaN in action"
        if jnp.any(jnp.isinf(action)):
            return False, "Inf in action"
        if action.shape != (8,):
            return False, f"Wrong action shape: {action.shape}"
        if jnp.any(jnp.abs(action) > 1.0 + 1e-6):
            return False, f"Action out of bounds: [{float(jnp.min(action)):.2f}, {float(jnp.max(action)):.2f}]"
            
        return True, "OK"
    except Exception as e:
        return False, f"Exception: {str(e)}"
