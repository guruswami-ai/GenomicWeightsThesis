#!/usr/bin/env python3
"""Simple distributed mesh test - just init and print rank"""
import mlx.core as mx
import socket

def main():
    print(f"[{socket.gethostname()}] Attempting MLX distributed init...")
    try:
        group = mx.distributed.init()
        rank = group.rank()
        size = group.size()
        print(f"[{socket.gethostname()}] SUCCESS! Rank {rank}/{size}")
    except Exception as e:
        print(f"[{socket.gethostname()}] FAILED: {e}")
        raise

if __name__ == "__main__":
    main()
