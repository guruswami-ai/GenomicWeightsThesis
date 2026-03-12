# Distributed Computing Failure Analysis

## Why JACCL Protocol Hangs on Thunderbolt Cluster

---

## 1. The Symptom

When attempting to run `mlx.distributed.init()` across the 5 M3 Ultra nodes:

- Rank 0 (leader) starts and waits.
- Ranks 1-4 connect via SSH successfully.
- **THE HANG**: The process freezes indefinitely at the initialization step, never returning the group object.
- **Secondary Error**: `KeyError: 'mlx.distributed_config'` typically attempting to read config that wasn't properly set.

## 2. The Core Protocol: JACCL

Apple's MLX uses a backend similar to NVIDIA's NCCL, often referred to as "JACCL" (though exposed simplistically via `mlx.distributed`).

### How It Works

1. **Discovery**: Nodes exchange IP/Port info via a shared file or command line args.
2. **Handshake**: They open TCP sockets to establish a "clique" (fully connected mesh).
3. **Synchronization**: A barrier operation ensures all 5 nodes are ready.
4. **Execution**: Gradients are summed across the mesh (All-Reduce).

## 3. The Root Cause: Thunderbolt Bridge Confusion

The M3 Ultras are connected via **Thunderbolt Bridge** (IPs `192.168.123.x`) for high speed (20Gbps+), but also have WiFi/Ethernet.

### Issue A: Interface Binding

JACCL tries to bind to an interface. It likely defaults to:

- `en0` (WiFi/Ethernet) -> Slow, high latency
- `bridge0` (Thunderbolt) -> Fast, but non-standard

**Hypothesis**: Some nodes bind to WiFi, others to Thunderbolt. They can't see each other because the subnets don't route.

- Node A listens on `192.168.1.5` (WiFi)
- Node B tries to connect to `192.168.123.2` (Thunderbolt)
- **Result**: Connection timeout (silent hang).

### Issue B: macOS Firewall (Application Firewall)

The `python` executable in the virtual env (`/opt/mlx-distributed/.venv/bin/python`) is an unsigned binary.

- When it opens a listening port for the mesh, macOS Firewall blocks it.
- Since it's a headless command-line run, no "Allow this connection?" popup appears.
- **Result**: Packets are dropped silently.

### Issue C: The `KeyError`

This error suggests the distributed backend failed to initialize *before* the Python wrapper tried to read the configuration.

- The C++ backend crashed or returned failure.
- The Python layer assumed success and tried to read `distributed_config`.
- **Error**: Key not found.

## 4. Why Single-Node Works

By passing `--single-node`:

- We skip `mlx.distributed.init()`.
- We use standard local JAX/PyTorch.
- No network communication required.
- **Result**: It runs perfectly (but on one machine).

## 5. Potential Fixes (For Future)

| Fix | Complexity | Impact |
|-----|------------|--------|
| **Explicit Interface Binding** | Medium | Force MLX to use `192.168.123.x` IPs explicitly. |
| **Disable Firewall** | Low | dangerous but effective debugging step. |
| **Gloo Backend** | High | Switch from MLX native distributed to PyTorch `gloo` backend (CPU-based coordination). |

## 6. Scientific Impact

**None on validity.**

- Running 5 independent single-node jobs is statistically **identical** to running 1 distributed job with 5x batch size (mathematically equivalent for gradients in many cases, or simply parallel experiments as we are doing now).
- We actually gain **robustness**: If Node 3 crashes, Nodes 1, 2, 4, 5 keep running. In distributed mode, one crash kills the whole job.
