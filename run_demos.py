#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для запуску демонстраційних ноутбуків для різних агентів
"""

import os
import sys
import argparse
import subprocess


def run_notebook(notebook_path):
    """Запустити Jupyter notebook з використанням nbconvert"""
    print(f"Виконання ноутбука: {notebook_path}")
    try:
        cmd = ["jupyter", "nbconvert", "--to", "notebook", "--execute", 
               "--inplace", "--ExecutePreprocessor.timeout=600", notebook_path]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Ноутбук успішно виконано")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Помилка при виконанні ноутбука: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Запуск демонстрацій для різних агентів")
    parser.add_argument("--ppo", action="store_true", help="Запустити PPO демонстрацію")
    parser.add_argument("--evolution", action="store_true", help="Запустити Evolution демонстрацію")
    parser.add_argument("--transformer", action="store_true", help="Запустити Transformer демонстрацію")
    parser.add_argument("--all", action="store_true", help="Запустити всі демонстрації")
    
    args = parser.parse_args()
    
    # Шлях до ноутбуків
    notebooks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
    ppo_notebook = os.path.join(notebooks_dir, "PPO_Demo.ipynb")
    evolution_notebook = os.path.join(notebooks_dir, "Evolution_Demo.ipynb")
    transformer_notebook = os.path.join(notebooks_dir, "Transformer_Demo.ipynb")
    
    # Перевірка, чи вибрано хоча б одну опцію
    if not (args.ppo or args.evolution or args.transformer or args.all):
        parser.print_help()
        return
    
    # Запуск вибраних демонстрацій
    if args.ppo or args.all:
        if os.path.exists(ppo_notebook):
            run_notebook(ppo_notebook)
        else:
            print(f"Не знайдено ноутбук: {ppo_notebook}")
    
    if args.evolution or args.all:
        if os.path.exists(evolution_notebook):
            if os.path.getsize(evolution_notebook) > 10:  # Перевірка, що ноутбук не порожній
                run_notebook(evolution_notebook)
            else:
                print(f"Ноутбук {evolution_notebook} здається порожнім і потребує доопрацювання")
        else:
            print(f"Не знайдено ноутбук: {evolution_notebook}")
    
    if args.transformer or args.all:
        if os.path.exists(transformer_notebook):
            if os.path.getsize(transformer_notebook) > 10:  # Перевірка, що ноутбук не порожній
                run_notebook(transformer_notebook)
            else:
                print(f"Ноутбук {transformer_notebook} здається порожнім і потребує доопрацювання")
        else:
            print(f"Не знайдено ноутбук: {transformer_notebook}")


if __name__ == "__main__":
    main() 