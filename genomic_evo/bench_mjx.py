import time
import jax
import jax.numpy as jnp
from fitness_env_mjx import mjx_step, mjx_model, mj_model, mujoco, mjx
from fitness_env import brax_ant_fitness # Legacy

def bench_mjx():
    print("🚀 Benchmarking MJX...")
    rng = jax.random.PRNGKey(0)
    
    # 1. Compilation Time
    start_time = time.time()
    jit_mjx_step = jax.jit(mjx_step).lower(None, rng).compile()
    compile_time = time.time() - start_time
    print(f"MJX Compile Time: {compile_time:.4f} s")
    
    # 2. Execution Time
    # Run once to ensure any runtime overhead is cleared
    jit_mjx_step(None, rng) 
    
    start_time = time.time()
    for _ in range(100):
        # We pass None as phenotype_net for this pure physics bench
        jit_mjx_step(None, rng)
        # Block to ensure execution finished
        jax.block_until_ready(rng) 
    
    duration = time.time() - start_time
    sps = (100 * 200) / duration # 100 calls * 200 steps per call
    print(f"MJX Throughput: {sps:.2f} steps/sec")

def bench_brax(phenotype_net_mock):
    print("\n🐢 Benchmarking Brax (Legacy)...")
    # Note: braiding into existing fitness_env is harder because it expects a net
    # We will just verify import speed and maybe a specialized small JIT test
    pass

if __name__ == "__main__":
    print(f"JAX Device: {jax.devices()[0]}")
    try:
        bench_mjx()
    except Exception as e:
        print(f"MJX Bench Failed: {e}")
