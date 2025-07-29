

import json
import os
import sys
import time
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import EvaluationMetrics
from react.react import run_react
from rag.main import run_rag
from finetuned_llm.inference import run_inference

class AskWPIEvaluator:
    def __init__(self):
        """Initialize the evaluator"""
        self.metrics = EvaluationMetrics()
        
    def load_test_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load test data from JSON file"""
        test_cases = []
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                test_cases = data.get('test_cases', [])
        except Exception as e:
            print(f"Error loading test data: {e}")
            return []
        
        return test_cases
    
    def run_single_prompt_evaluation(self, model_name: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run single prompt evaluation for a model"""
        print(f"Running Single Prompt Evaluation for {model_name.upper()}")
        
        predictions = []
        ground_truths = []
        failed_cases = []
        
        for i, test_case in enumerate(test_cases):
            try:
                # Extract query and ground truth
                query = test_case['query']
                ground_truth = test_case['ground_truth']
                
                print(f"Processing test case {i+1}/{len(test_cases)}: {query[:50]}...")
                
                # Get model prediction
                start_time = time.time()
                
                if model_name == "rag":
                    prediction = run_rag(query)
                elif model_name == "react":
                    prediction = run_react(query)
                elif model_name == "finetuned":
                    prediction = run_inference(query)
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
                end_time = time.time()
                
                predictions.append(prediction)
                ground_truths.append(ground_truth)
                
                print(f"  ✓ Completed in {end_time - start_time:.2f}s")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                failed_cases.append({
                    'index': i,
                    'query': query if 'query' in locals() else 'Unknown',
                    'error': str(e)
                })
                # Add empty prediction to maintain alignment
                predictions.append("")
                ground_truths.append(ground_truth if 'ground_truth' in locals() else "")
        
        # Calculate metrics
        metrics = self.metrics.evaluate_batch(predictions, ground_truths)
        
        results = {
            'model': model_name,
            'evaluation_type': 'single_prompt',
            'total_cases': len(test_cases),
            'successful_cases': len(test_cases) - len(failed_cases),
            'failed_cases': len(failed_cases),
            'failed_case_details': failed_cases,
            'metrics': metrics,
            'predictions': predictions,
            'ground_truths': ground_truths,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def run_multi_turn_evaluation(self, model_name: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run multi-turn evaluation for a model"""
        print(f"Running Multi-Turn Evaluation for {model_name.upper()}")
        
        conversation_responses = []
        ground_truths = []
        failed_cases = []
        
        for i, test_case in enumerate(test_cases):
            try:
                # Create a simple multi-turn scenario from single test case
                query = test_case['query']
                ground_truth = test_case['ground_truth']
                
                print(f"Processing multi-turn test case {i+1}/{len(test_cases)}")
                
               
                turn1_query = query
                turn2_query = f"Can you provide more details about {query.lower().split()[0]}?"
                
                conversation = []
                gt_responses = []
                
                # Process Turn 1
                print(f"  Turn 1: {turn1_query[:30]}...")
                start_time = time.time()
                
                if model_name == "rag":
                    prediction1 = run_rag(turn1_query)
                elif model_name == "react":
                    prediction1 = run_react(turn1_query)
                elif model_name == "finetuned":
                    prediction1 = run_inference(turn1_query)
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
                end_time = time.time()
                conversation.append(prediction1)
                gt_responses.append(ground_truth)
                print(f"    ✓ Completed in {end_time - start_time:.2f}s")
                
                # Process Turn 2
                print(f"  Turn 2: {turn2_query[:30]}...")
                start_time = time.time()
                
                if model_name == "rag":
                    prediction2 = run_rag(turn2_query)
                elif model_name == "react":
                    prediction2 = run_react(turn2_query)
                elif model_name == "finetuned":
                    prediction2 = run_inference(turn2_query)
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
                end_time = time.time()
                conversation.append(prediction2)
                gt_responses.append(ground_truth)  # Use same ground truth for simplicity
                print(f"    ✓ Completed in {end_time - start_time:.2f}s")
                
                conversation_responses.append(conversation)
                ground_truths.append(gt_responses)
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                failed_cases.append({
                    'index': i,
                    'error': str(e)
                })
        
        # Calculate multi-turn metrics
        if conversation_responses:
            metrics = self.metrics.evaluate_multi_turn(conversation_responses, ground_truths)
        else:
            metrics = {
                'multi_turn_consistency': 0.0,
                'exact_match': 0.0,
                'semantic_accuracy': 0.0,
                'f1_score': 0.0,
                'consistency_std': 0.0
            }
        
        results = {
            'model': model_name,
            'evaluation_type': 'multi_turn',
            'total_cases': len(test_cases),
            'successful_cases': len(test_cases) - len(failed_cases),
            'failed_cases': len(failed_cases),
            'failed_case_details': failed_cases,
            'metrics': metrics,
            'conversation_responses': conversation_responses,
            'ground_truths': ground_truths,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def evaluate_all_models(self, test_data_path: str, output_dir: str = "evaluation_results"):
        """Evaluate all three models"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data
        print(f"Loading test data from {test_data_path}")
        test_cases = self.load_test_data(test_data_path)
        
        if not test_cases:
            print("No test cases loaded. Exiting.")
            return
        
        print(f"Loaded {len(test_cases)} test cases")
        
        # Models to evaluate
        models = ["rag", "react", "finetuned"]
        
        all_results = {}
        
        for model in models:
            print(f"\n{'='*60}")
            print(f"EVALUATING {model.upper()} MODEL")
            print(f"{'='*60}")
            
            # Single prompt evaluation
            single_results = self.run_single_prompt_evaluation(model, test_cases)
            all_results[f"{model}_single"] = single_results
            
            # Multi-turn evaluation
            multi_results = self.run_multi_turn_evaluation(model, test_cases)
            all_results[f"{model}_multi"] = multi_results
            
            # Save individual model results
            model_output_file = os.path.join(output_dir, f"{model}_evaluation_results.json")
            with open(model_output_file, 'w') as f:
                json.dump({
                    'single_prompt': single_results,
                    'multi_turn': multi_results
                }, f, indent=2)
            
            print(f"Results saved to {model_output_file}")
        
        # Generate comparison report
        self.generate_comparison_report(all_results, output_dir)
        
        # Save all results
        all_results_file = os.path.join(output_dir, "all_evaluation_results.json")
        with open(all_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nAll results saved to {all_results_file}")
        
        return all_results
    
    def generate_comparison_report(self, results: Dict[str, Any], output_dir: str):
        """Generate a comparison report of all models"""
        print(f"\n{'='*60}")
        print("GENERATING COMPARISON REPORT")
        print(f"{'='*60}")
        
        # Extract metrics for comparison
        comparison_data = []
        
        for key, result in results.items():
            model_name = key.split('_')[0]
            eval_type = key.split('_')[1]
            
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name.upper(),
                'Evaluation Type': eval_type.replace('_', ' ').title(),
                'Exact Match': f"{metrics['exact_match']:.4f}",
                'Semantic Accuracy': f"{metrics['semantic_accuracy']:.4f}",
                'F1 Score': f"{metrics['f1_score']:.4f}",
                'Multi-turn Consistency': f"{metrics.get('multi_turn_consistency', 'N/A')}",
                'Successful Cases': result['successful_cases'],
                'Failed Cases': result['failed_cases']
            })
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(comparison_data)
        csv_file = os.path.join(output_dir, "model_comparison.csv")
        df.to_csv(csv_file, index=False)
        
        # Print comparison table
        print("\nModel Comparison:")
        print(df.to_string(index=False))
        
       
        print(f"Comparison table saved to {csv_file}")
        
        return df

def main():
    """Main function to run evaluation"""
    evaluator = AskWPIEvaluator()
    
    # Use the training dataset as test data
    test_data_path = "finetuned_llm/dataset.jsonl"
    
    if not os.path.exists(test_data_path):
        print(f"Test data file not found: {test_data_path}")
        return
    
    # Run evaluation
    results = evaluator.evaluate_all_models(test_data_path)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETED")
    print(f"{'='*60}")
    print("Check the 'evaluation_results' directory for detailed results.")

if __name__ == "__main__":
    main() 