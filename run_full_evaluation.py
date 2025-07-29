#!/usr/bin/env python3
"""
Main script to run the complete AskWPI model evaluation
"""

import os
import sys
from evaluation.evaluator import AskWPIEvaluator

def main():
    """Run the complete evaluation"""
    print("AskWPI Complete Model Evaluation")
    print("=" * 50)
    print("This will evaluate all three models (RAG, ReACT, Fine-tuned LLM)")
    print("using the training dataset as test data.")
    print("=" * 50)
    
    # Check if test data exists
    test_data_path = "evaluation/test_data.json"
    
    if not os.path.exists(test_data_path):
        print(f"Error: Test data file not found at {test_data_path}")
        print("Please ensure the test_data.json file exists in the evaluation directory.")
        return
    
    print(f"Using test data from: {test_data_path}")
    
    # Check dataset size
    with open(test_data_path, 'r') as f:
        import json
        data = json.load(f)
        num_lines = len(data.get('test_cases', []))
    print(f"Dataset contains {num_lines} test cases")
    
    # Estimate time
    estimated_time = (num_lines * 3 * 3) / 60  # 3 models, 3 seconds per case
    print(f"Estimated evaluation time: {estimated_time:.1f} minutes")
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with the evaluation? (y/N): ")
    if response.lower() != 'y':
        print("Evaluation cancelled.")
        return
    
    # Initialize evaluator and run evaluation
    try:
        print(f"\nStarting evaluation...")
        evaluator = AskWPIEvaluator()
        results = evaluator.evaluate_all_models(test_data_path)
        
        if results:
            print(f"\n{'='*50}")
            print("EVALUATION COMPLETED SUCCESSFULLY")
            print(f"{'='*50}")
            print("Results saved to 'evaluation_results' directory")
            print("Files generated:")
            print("   • all_evaluation_results.json")
            print("   • model_comparison.csv")
            print("   • summary_statistics.json")
            print("   • [model]_evaluation_results.json")
            
            # Print quick summary
            print(f"\nQuick Summary:")
            for key, result in results.items():
                if 'single' in key:
                    model = key.split('_')[0].upper()
                    metrics = result['metrics']
                    print(f"   {model}:")
                    print(f"     • Exact Match: {metrics['exact_match']:.4f}")
                    print(f"     • Semantic Accuracy: {metrics['semantic_accuracy']:.4f}")
                    print(f"     • F1 Score: {metrics['f1_score']:.4f}")
                    print(f"     • Success Rate: {result['successful_cases']}/{result['total_cases']}")
                    print()
        else:
            print(" Evaluation failed. Check the error messages above.")
            
    except KeyboardInterrupt:
        print("\n  Evaluation interrupted by user.")
    except Exception as e:
        print(f"\n Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()