import mlx.core as mx
import os

def main():
    rank = mx.distributed.init().rank()
    hostname = os.uname().nodename
    print(f"Hello from rank {rank} on {hostname}")

if __name__ == "__main__":
    main()
