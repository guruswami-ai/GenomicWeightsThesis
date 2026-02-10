"""
MJX (MuJoCo XLA) fitness environment for the Genomic Weight Thesis experiment.

Note: MJX doesn't fully support JAX Metal (Apple Silicon GPU) yet.
We explicitly use CPU device which is still fast on M3 Ultra (~650K steps/sec).
See: https://github.com/jax-ml/jax/issues/26968
"""
# MUST be set before any JAX import to force CPU backend
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import jit, vmap
import mujoco
from mujoco import mjx

# Standard Ant XML (Matches Brax Ant-v4 physics)
ANT_XML = """
<mujoco model="ant">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <custom>
    <numeric data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 0.75">
      <geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
      <body name="front_left_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="aux_1_geom" size="0.08" type="capsule"/>
        <body name="aux_1" pos="0.2 0.2 0">
          <joint axis="0 0 1" name="hip_1" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="left_leg_geom" size="0.08" type="capsule"/>
          <body name="left_leg" pos="0.2 0.2 0">
            <joint axis="-1 1 0" name="ankle_1" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.4 0.4 0.0" name="left_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="front_right_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="aux_2_geom" size="0.08" type="capsule"/>
        <body name="aux_2" pos="-0.2 0.2 0">
          <joint axis="0 0 1" name="hip_2" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="right_leg_geom" size="0.08" type="capsule"/>
          <body name="right_leg" pos="-0.2 0.2 0">
            <joint axis="1 1 0" name="ankle_2" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -0.4 0.4 0.0" name="right_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="back_left_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="aux_3_geom" size="0.08" type="capsule"/>
        <body name="aux_3" pos="-0.2 -0.2 0">
          <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="back_leg_geom" size="0.08" type="capsule"/>
          <body name="back_leg" pos="-0.2 -0.2 0">
            <joint axis="-1 1 0" name="ankle_3" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -0.4 -0.4 0.0" name="back_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="back_right_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 0.2 -0.2 0.0" name="aux_4_geom" size="0.08" type="capsule"/>
        <body name="aux_4" pos="0.2 -0.2 0">
          <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.2 -0.2 0.0" name="rightback_leg_geom" size="0.08" type="capsule"/>
          <body name="rightback_leg" pos="0.2 -0.2 0">
            <joint axis="1 1 0" name="ankle_4" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.4 -0.4 0.0" name="rightback_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_3" gear="150"/>
  </actuator>
</mujoco>
"""

# Global cache - initialized on first use
_MJ_MODEL = None
_MJX_MODEL = None
_MJX_DATA_TEMPLATE = None
_CPU_DEVICE = None


def _get_cpu_device():
    """Get CPU device (cached)."""
    global _CPU_DEVICE
    if _CPU_DEVICE is None:
        _CPU_DEVICE = jax.devices("cpu")[0]
    return _CPU_DEVICE


def get_model():
    """Load MuJoCo Ant model and put on CPU device for MJX.

    MJX doesn't fully support JAX Metal (Apple Silicon GPU) yet,
    so we explicitly use CPU which is still fast for M3 Ultra.
    """
    global _MJ_MODEL, _MJX_MODEL, _MJX_DATA_TEMPLATE
    if _MJ_MODEL is None:
        cpu_device = _get_cpu_device()
        _MJ_MODEL = mujoco.MjModel.from_xml_string(ANT_XML)
        _MJX_MODEL = mjx.put_model(_MJ_MODEL, device=cpu_device)
        # Pre-create data template for reuse
        data = mujoco.MjData(_MJ_MODEL)
        _MJX_DATA_TEMPLATE = mjx.put_data(_MJ_MODEL, data, device=cpu_device)
    return _MJ_MODEL, _MJX_MODEL, _MJX_DATA_TEMPLATE


def get_obs(data):
    """
    Extract Brax-compatible observation (27-dim).
    [qpos[2:] (z + quat + joints = 13) + qvel (14) = 27 dims]
    """
    return jnp.concatenate([data.qpos[2:], data.qvel])


@jax.jit
def _rollout_jit(mjx_model, dx, rng, action_fn_output, episode_length=200):
    """
    JIT-compiled episode rollout. Action is computed outside JIT and passed in.
    This allows the JIT to be cached across different phenotype networks.
    """
    def step_fn(carry, action):
        dx, key = carry

        # Apply action
        action = jnp.clip(action, -1.0, 1.0)
        dx = dx.replace(ctrl=action)
        dx = mjx.step(mjx_model, dx)

        # Reward: forward velocity + survival bonus
        reward = dx.qvel[0] + 1.0

        return (dx, key), reward

    # Run rollout with pre-computed actions
    (final_dx, _), rewards = jax.lax.scan(step_fn, (dx, rng), action_fn_output, length=episode_length)

    return jnp.sum(rewards)


def _rollout_with_policy(phenotype_net, mjx_model, dx, rng, episode_length=200):
    """
    Run a rollout by stepping through and computing actions with phenotype_net.
    Non-JIT version that works with any phenotype_net.
    """
    total_reward = 0.0
    for _ in range(episode_length):
        obs = get_obs(dx)
        action = phenotype_net(obs)
        action = jnp.clip(action, -1.0, 1.0)
        dx = dx.replace(ctrl=action)
        dx = mjx.step(mjx_model, dx)
        total_reward += dx.qvel[0] + 1.0
    return total_reward


def brax_ant_fitness(phenotype_net, phenotype_data=None, num_rollouts=2):
    """
    Evaluate phenotype network on Ant locomotion task using MJX.

    Uses MJX on CPU (JAX Metal doesn't fully support MJX yet).

    Args:
        phenotype_net: Callable that maps obs (27,) -> action (8,)
        phenotype_data: Optional dict (unused, for API compatibility)
        num_rollouts: Number of independent rollouts to average

    Returns:
        Mean reward across rollouts
    """
    mj_model, mjx_model, data_template = get_model()
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, num_rollouts)

    # Run rollouts (using simple loop - JIT compilation is tricky with closures)
    rewards = []
    for i in range(num_rollouts):
        dx = jax.tree.map(lambda x: x, data_template)
        reward = _rollout_with_policy(phenotype_net, mjx_model, dx, rngs[i])
        rewards.append(reward)

    return jnp.mean(jnp.array(rewards))


# Alias for backward compatibility
predator_avoidance_fitness = brax_ant_fitness
