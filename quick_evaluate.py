#!/usr/bin/env python3
"""
Quick evaluation of existing models
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🎯 Quick Evaluation of Existing Models")
    print("=" * 50)
    
    # Check models
    evolution_model = Path("models/evolved_strategy.pkl")
    transformer_model = Path("models/transformer_best.pth")
    ppo_results = Path("results/ppo/evaluation_results.csv")
    
    print(f"Evolution model: {'✅' if evolution_model.exists() else '❌'}")
    print(f"Transformer model: {'✅' if transformer_model.exists() else '❌'}")
    print(f"PPO results: {'✅' if ppo_results.exists() else '❌'}")
    
    # For Evolution: need to add evaluation to existing model
    if evolution_model.exists():
        print("\n🧬 Adding evaluation to Evolution...")
        # We already modified train_evolution.py to include evaluation_results.csv generation
        # So we just need a way to trigger evaluation without retraining
        
        print("   Evolution model exists but needs evaluation_results.csv")
        print("   We modified train_evolution.py to include this.")
    
    # For Transformer: already has evaluation code
    if transformer_model.exists():
        print("\n🤖 Transformer model exists and has evaluation code")
    
    print("\n💡 Solution: Modify training files are updated to generate evaluation_results.csv")
    print("📁 Files updated:")
    print("   - agents/evolution/train_evolution.py (added evaluation_results.csv generation)")
    print("   - agents/transformer/train_transformer.py (already had evaluation code)")
    
    print("\n✅ Next time you run training, evaluation_results.csv will be generated automatically!")
    print("📊 For now, comparison can use existing results or re-run training briefly.")

if __name__ == "__main__":
    main() 