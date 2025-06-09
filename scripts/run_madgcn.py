import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basicts import launch_training

def run_madgcn(config_file=None, gpus=None, seed=None):
    if config_file is None:
        config_file = 'model/MADGCN/LargeAQ.py'
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Starting MADGCN training with config: {config_file}")
    if gpus:
        print(f"Using GPU(s): {gpus}")
    if seed:
        print(f"Random seed: {seed}")
    launch_training(config_file, gpus=gpus)

def main():
    parser = argparse.ArgumentParser(description='Run MADGCN training')
    parser.add_argument('--config_file', type=str, default='model/MADGCN/LargeAQ.py',
                        help='Path to config file')
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU devices to use')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    args = parser.parse_args()
    run_madgcn(args.config_file, args.gpus, args.seed)

if __name__ == '__main__':
    main() 