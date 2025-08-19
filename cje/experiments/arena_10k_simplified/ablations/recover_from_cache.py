#!/usr/bin/env python3
"""Recover experiment results from cache files."""

import json
from pathlib import Path
import numpy as np

def convert_numpy(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(v) for v in obj)
    return obj


def recover_ablation_results(ablation_name: str):
    """Recover results from cache for a given ablation."""
    
    cache_dir = Path(f"../.ablation_cache/{ablation_name}")
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return []
    
    results = []
    cache_files = list(cache_dir.glob("*.json"))
    
    print(f"Found {len(cache_files)} cached results for {ablation_name}")
    
    for cache_file in cache_files:
        try:
            with open(cache_file) as f:
                result = json.load(f)
                # Convert any remaining numpy types
                result = convert_numpy(result)
                results.append(result)
        except Exception as e:
            print(f"Error reading {cache_file}: {e}")
    
    return results


def main():
    """Recover all ablation results from cache."""
    
    print("=" * 70)
    print("RECOVERING RESULTS FROM CACHE")
    print("=" * 70)
    print()
    
    ablations = ["oracle_coverage", "sample_size", "interaction"]
    
    for ablation in ablations:
        print(f"\nRecovering {ablation}...")
        results = recover_ablation_results(ablation)
        
        if results:
            # Save to results directory
            output_dir = Path(f"ablations/results/{ablation}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / "results.jsonl"
            
            # Back up existing file if it exists
            if output_file.exists():
                backup = output_dir / "results.jsonl.bak"
                output_file.rename(backup)
                print(f"  Backed up existing results to {backup}")
            
            # Write recovered results
            with open(output_file, "w") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")
            
            # Count successful results
            successful = sum(1 for r in results if r.get("success", False))
            print(f"  Saved {len(results)} results ({successful} successful) to {output_file}")
            
            # Show summary
            if successful > 0:
                # Group by configuration
                configs = {}
                for r in results:
                    if r.get("success", False):
                        spec = r["spec"]
                        key = f"{spec.get('oracle_coverage', 'N/A')}_{spec.get('sample_size', 'N/A')}_{spec.get('estimator', 'N/A')}"
                        if key not in configs:
                            configs[key] = []
                        configs[key].append(r.get("rmse_vs_oracle", np.nan))
                
                print(f"  Configurations: {len(configs)}")
        else:
            print(f"  No results found in cache")
    
    print()
    print("=" * 70)
    print("RECOVERY COMPLETE")
    print("=" * 70)
    print("\nNow run: python analyze_results.py --figures")


if __name__ == "__main__":
    main()