"""
Main script to run the emotion classification evaluation
Provides a command-line interface to run different evaluation experiments
"""
import os
import sys
import traceback
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from results.model_evaluation import main

if __name__ == "__main__":
    try:
        print("Starting emotion classification model evaluation...")
        main()
        print("Evaluation completed successfully!")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        sys.exit(1)
