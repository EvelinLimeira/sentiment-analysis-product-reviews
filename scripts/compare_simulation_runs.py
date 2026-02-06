"""
Compare results from different simulation runs.

This script helps you compare multiple simulation runs without overwriting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from datetime import datetime

def list_available_runs():
    """List all available simulation runs."""
    simulations_dir = Path('results/simulations')
    runs = []
    
    # Check for timestamped runs
    for run_dir in simulations_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith('run_'):
            metadata_file = run_dir / 'run_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                runs.append({
                    'run_id': run_dir.name,
                    'path': run_dir,
                    'metadata': metadata
                })
    
    return runs

def compare_runs(run_ids=None):
    """Compare multiple simulation runs."""
    runs = list_available_runs()
    
    if not runs:
        print("No timestamped runs found in results/simulations/")
        print("\nChecking for default run...")
        # Check default location
        default_files = list(Path('results/simulations').glob('*_simulations.csv'))
        if default_files:
            print(f"Found {len(default_files)} files in default location")
            return
        else:
            print("No simulation results found.")
            return
    
    print("=" * 80)
    print("AVAILABLE SIMULATION RUNS")
    print("=" * 80)
    print()
    
    for i, run in enumerate(runs, 1):
        metadata = run['metadata']
        print(f"{i}. Run ID: {run['run_id']}")
        print(f"   Timestamp: {metadata.get('timestamp', 'N/A')}")
        print(f"   Simulations: {metadata.get('num_simulations', 'N/A')}")
        print(f"   Models: {', '.join(metadata.get('models', []))}")
        print(f"   Path: {run['path']}")
        print()
    
    # If specific runs requested, compare them
    if run_ids:
        print("=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print()
        
        for run_id in run_ids:
            run_path = Path(f'results/simulations/{run_id}')
            if not run_path.exists():
                print(f"Run not found: {run_id}")
                continue
            
            print(f"\n{run_id}:")
            print("-" * 80)
            
            # Load and display results
            for csv_file in run_path.glob('*_simulations.csv'):
                model_name = csv_file.stem.replace('_simulations', '')
                df = pd.read_csv(csv_file)
                
                print(f"\n{model_name.upper()}:")
                if 'f1_macro' in df.columns:
                    print(f"  F1-Score: {df['f1_macro'].mean():.4f} ± {df['f1_macro'].std():.4f}")
                elif 'f1_score' in df.columns:
                    print(f"  F1-Score: {df['f1_score'].mean():.4f} ± {df['f1_score'].std():.4f}")
                
                if 'accuracy' in df.columns:
                    print(f"  Accuracy: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare simulation runs')
    parser.add_argument('--runs', nargs='+', help='Specific run IDs to compare')
    parser.add_argument('--list', action='store_true', help='List all available runs')
    
    args = parser.parse_args()
    
    if args.list or not args.runs:
        compare_runs()
    else:
        compare_runs(args.runs)

if __name__ == '__main__':
    main()
