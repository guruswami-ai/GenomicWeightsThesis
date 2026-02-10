#!/usr/bin/env python3
"""Quick test of all three strategies to verify architecture works"""
import jax
import jax.numpy as jnp
from genotype_nets import FlatCompressor, HierarchicalCompressor, TopologicalCompressor
from phenotype_net import PhenotypeNet
from jax.flatten_util import ravel_pytree

def test_strategy(strategy_name, net_class):
    """Test a single strategy end-to-end"""
    print(f"\n{'='*50}")
    print(f"Testing: {strategy_name}")
    print(f"{'='*50}")
    
    # Create genotype network
    z = jnp.zeros((1, 128))
    net = net_class()
    
    # Initialize and forward
    variables = net.init(jax.random.PRNGKey(0), z)
    flat_params, unflatten_fn = ravel_pytree(variables)
    print(f"  Genotype params: {len(flat_params):,}")
    
    # Get phenotype data
    p_data = net.apply(variables, z)
    print(f"  Phenotype data keys: {list(p_data.keys())}")
    
    # Create phenotype network and test forward pass
    p_net = PhenotypeNet(phenotype_data=p_data)
    
    # Test with Brax-like observation (27 dims)
    obs = jnp.ones(27) * 0.5
    
    # Call the phenotype network
    action = p_net.apply({}, obs)  # Empty params - uses genotype-emitted weights
    
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action shape: {action.shape}")
    print(f"  Action values: {action}")
    print(f"  ✅ {strategy_name} PASSED!")
    
    return True

def main():
    print("Testing Genomic Compression Architecture")
    print("="*60)
    
    results = {}
    
    # Test each strategy
    for name, net_class in [
        ("Flat", FlatCompressor),
        ("Hierarchical", HierarchicalCompressor),
        ("Topological", TopologicalCompressor),
    ]:
        try:
            results[name] = test_strategy(name, net_class)
        except Exception as e:
            print(f"  ❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    print("\n" + "="*60)
    print("SUMMARY:")
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    if all(results.values()):
        print("\n🎉 All strategies working! Ready for comparative pilots.")
    else:
        print("\n⚠️ Some strategies need fixing.")
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
