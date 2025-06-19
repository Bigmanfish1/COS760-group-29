"""
Run evaluation of fine-tuned mBERT and AfriBERTa models
on Zulu and Nigerian Pidgin English test sets from the BRIGHTER dataset.
"""
import os
import sys
import traceback
import argparse

# Add the current directory to Python path to make local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the evaluation function from the results package
from results.evaluate_fine_tuned import main

if __name__ == "__main__":
    try:
        print("Starting evaluation of fine-tuned emotion classification models...")
        parser = argparse.ArgumentParser(description="Evaluate fine-tuned models on test datasets")
        
        args = parser.parse_args()
        
        # Run the evaluation
        main()
        
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        sys.exit(1)
