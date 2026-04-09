import argparse
from training.train import train
from evaluation.evaluate import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "eval":
        evaluate()

if __name__ == "__main__":
    main()