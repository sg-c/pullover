import sys
from src.train import train

def main():
    if len(sys.argv) < 2:
        print("Usage: python po.py <command>")
        print("Available commands: train")
        return

    command = sys.argv[1]

    if command == "train":
        print("Starting training...")
        train()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train")

if __name__ == "__main__":
    main()
