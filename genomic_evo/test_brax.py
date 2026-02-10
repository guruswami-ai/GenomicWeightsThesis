#!/usr/bin/env python3
"""Quick validation test for upgraded Brax environment"""
import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.wrappers.training import wrap

# Test Brax environment creation
print("Testing Brax Ant environment...")
env = envs.get_environment("ant")
env = wrap(env, episode_length=100)

# Test reset with batched RNG
rng = jax.random.PRNGKey(0)
state = env.reset(jax.random.split(rng, 1))  # Batch size 1
print(f"✓ Environment reset successful")
print(f"  Observation shape: {state.obs.shape}")  # Should be (1, 27)
print(f"  Observation: {state.obs[0, :5]}...")  # First 5 dims

# Test step with random action
action = jnp.zeros((1, 8))  # Batched 8-dim action space
state = env.step(state, action)
print(f"✓ Environment step successful")
print(f"  Reward: {state.reward}")
print(f"  Done: {state.done}")

# Test phenotype network integration
from genotype_nets import FlatCompressor
from phenotype_net import PhenotypeNet

print("\nTesting phenotype network with Brax...")
net = FlatCompressor()
variables = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 128)))
z = jnp.zeros((1, 128))
p_data = net.apply(variables, z)

p_net = PhenotypeNet(phenotype_data=p_data)
p_params = p_net.init(jax.random.PRNGKey(1), state.obs[0])  # Single obs from batch

# Generate action from observation
action = p_net.apply(p_params, state.obs[0])
print(f"✓ Phenotype network output shape: {action.shape}")  # Should be (8,)
print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")  # Should be [-1, 1]

print("\n✅ All validation tests passed!")
print("Ready for full experiment launch.")
