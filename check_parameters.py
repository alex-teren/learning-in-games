#!/usr/bin/env python3
"""
Parameter Unification Check for IPD Approaches
Verifies that all three approaches (PPO, Evolution, Transformer) use consistent parameters
"""

import ast
import re
from pathlib import Path


def extract_defaults_from_file(file_path: str) -> dict:
    """Extract default parameters from Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        defaults = {}
        
        # Find num_rounds defaults
        num_rounds_matches = re.findall(r'num_rounds.*?=.*?(\d+)', content)
        if num_rounds_matches:
            defaults['num_rounds'] = [int(x) for x in num_rounds_matches]
        
        # Find seed defaults  
        seed_matches = re.findall(r'seed.*?=.*?(\d+)', content)
        if seed_matches:
            defaults['seed'] = [int(x) for x in seed_matches]
        
        # Find strategy imports
        import_line_match = re.search(r'from env import.*?(TitForTat.*?GTFTStrategy|IPDEnv.*?GTFTStrategy)', content, re.DOTALL)
        if import_line_match:
            import_line = import_line_match.group(1)
            strategy_imports = []
            for strategy in ['TitForTat', 'AlwaysCooperate', 'AlwaysDefect', 'RandomStrategy', 'PavlovStrategy', 'GrudgerStrategy', 'GTFTStrategy']:
                if strategy in import_line:
                    strategy_imports.append(strategy)
            defaults['imported_strategies'] = strategy_imports
        else:
            defaults['imported_strategies'] = []
        
        # Find strategy usage in lists
        strategy_lists = re.findall(r'\[(.*?)\]', content.replace('\n', ' '))
        used_strategies = []
        for strategy_list in strategy_lists:
            for strategy in ['TitForTat', 'AlwaysCooperate', 'AlwaysDefect', 'RandomStrategy', 'PavlovStrategy', 'GrudgerStrategy', 'GTFTStrategy']:
                if strategy in strategy_list and strategy not in used_strategies:
                    used_strategies.append(strategy)
        
        defaults['used_strategies'] = used_strategies
        
        return defaults
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}


def check_unified_parameters():
    """Check parameter unification across all approaches"""
    
    print("🔍 Checking Parameter Unification for IPD Project")
    print("=" * 60)
    
    # File paths
    files = {
        'PPO': 'agents/ppo/train_ppo.py',
        'Evolution': 'agents/evolution/train_evolution.py', 
        'Transformer': 'agents/transformer/train_transformer.py'
    }
    
    # Extract parameters from each file
    all_params = {}
    for approach, file_path in files.items():
        print(f"\n📋 Analyzing {approach}: {file_path}")
        params = extract_defaults_from_file(file_path)
        all_params[approach] = params
        
        print(f"   num_rounds defaults: {params.get('num_rounds', [])}")
        print(f"   seed defaults: {params.get('seed', [])}")
        print(f"   imported strategies: {len(params.get('imported_strategies', []))}")
        print(f"   used strategies: {len(params.get('used_strategies', []))}")
    
    # Check unification
    print(f"\n🎯 UNIFICATION CHECK")
    print("=" * 60)
    
    # Check num_rounds consistency
    num_rounds_consistent = True
    ppo_num_rounds = set(all_params['PPO'].get('num_rounds', []))
    evolution_num_rounds = set(all_params['Evolution'].get('num_rounds', []))
    transformer_num_rounds = set(all_params['Transformer'].get('num_rounds', []))
    
    if not (100 in ppo_num_rounds and 100 in evolution_num_rounds and 100 in transformer_num_rounds):
        num_rounds_consistent = False
    
    print(f"✅ num_rounds=100 default: {'CONSISTENT' if num_rounds_consistent else '❌ INCONSISTENT'}")
    
    # Check seed consistency
    seed_consistent = True
    ppo_seeds = set(all_params['PPO'].get('seed', []))
    evolution_seeds = set(all_params['Evolution'].get('seed', []))
    transformer_seeds = set(all_params['Transformer'].get('seed', []))
    
    if not (42 in ppo_seeds and 42 in evolution_seeds and 42 in transformer_seeds):
        seed_consistent = False
        
    print(f"✅ seed=42 default: {'CONSISTENT' if seed_consistent else '❌ INCONSISTENT'}")
    
    # Check strategy imports
    expected_strategies = {'TitForTat', 'AlwaysCooperate', 'AlwaysDefect', 'RandomStrategy', 'PavlovStrategy', 'GrudgerStrategy', 'GTFTStrategy'}
    
    strategies_consistent = True
    for approach in ['PPO', 'Evolution', 'Transformer']:
        imported = set(all_params[approach].get('imported_strategies', []))
        if not expected_strategies.issubset(imported):
            strategies_consistent = False
            missing = expected_strategies - imported
            print(f"❌ {approach} missing imports: {missing}")
    
    print(f"✅ All 7 strategies imported: {'CONSISTENT' if strategies_consistent else '❌ INCONSISTENT'}")
    
    # Check strategy usage
    usage_consistent = True
    for approach in ['PPO', 'Evolution', 'Transformer']:
        used = set(all_params[approach].get('used_strategies', []))
        if len(used) < 7:
            usage_consistent = False
            
    print(f"✅ All 7 strategies used: {'CONSISTENT' if usage_consistent else '❌ INCONSISTENT'}")
    
    # Summary
    print(f"\n📊 OVERALL STATUS")
    print("=" * 60)
    
    all_consistent = num_rounds_consistent and seed_consistent and strategies_consistent and usage_consistent
    
    if all_consistent:
        print("🎉 ALL PARAMETERS UNIFIED! Ready for fair comparison.")
    else:
        print("⚠️  PARAMETERS NOT FULLY UNIFIED! Requires fixes before retraining.")
        
        print(f"\n🔧 Required Actions:")
        if not num_rounds_consistent:
            print("   - Ensure all approaches use num_rounds=100 as default")
        if not seed_consistent:
            print("   - Ensure all approaches use seed=42 as default") 
        if not strategies_consistent:
            print("   - Add missing strategy imports")
        if not usage_consistent:
            print("   - Update evaluation to use all 7 strategies")
    
    return all_consistent


if __name__ == "__main__":
    check_unified_parameters() 