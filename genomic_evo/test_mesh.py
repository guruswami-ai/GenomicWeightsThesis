import mlx.core as mx
import socket
import os

def main():
    # Explicit override for debugging
    if 'MLX_COORD_IP' not in os.environ:
        os.environ['MLX_COORD_IP'] = '10.61.106.31'
    if 'MLX_WORLD_SIZE' not in os.environ:
        os.environ['MLX_WORLD_SIZE'] = '5'
        
    rank_env = os.environ.get('MLX_RANK', 'Not Set')
    print(f"[{socket.gethostname()}] Starting init. Env MLX_RANK: {rank_env}")
    
    try:
        group = mx.distributed.init()
        rank = group.rank()
        size = group.size()
        print(f"[{socket.gethostname()}] init DONE. Rank: {rank}/{size}")
    except Exception as e:
        print(f"[{socket.gethostname()}] init FAILED: {e}")

if __name__ == "__main__":
    main()
