#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Утиліти для роботи з Jupyter ноутбуками
"""

import os
import sys
import argparse
import subprocess
import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def convert_notebook_to_python(notebook_path, output_dir=None):
    """Конвертувати Jupyter notebook у Python скрипт"""
    if not output_dir:
        output_dir = os.path.dirname(notebook_path)
    
    base_name = os.path.splitext(os.path.basename(notebook_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.py")
    
    print(f"Конвертація {notebook_path} в {output_path}")
    
    try:
        cmd = ["jupyter", "nbconvert", "--to", "python", 
               notebook_path, "--output", output_path]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Конвертація успішна: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Помилка конвертації: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None


def clear_notebook_output(notebook_path):
    """Очистити виведення комірок у ноутбуці"""
    print(f"Очищення виводу в {notebook_path}")
    
    try:
        cmd = ["jupyter", "nbconvert", "--clear-output", "--inplace", notebook_path]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Ноутбук успішно очищено: {notebook_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Помилка при очищенні ноутбука: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False


def create_notebook_template(output_path, title, description, imports=[]):
    """
    Створити шаблон Jupyter ноутбука
    
    Args:
        output_path: шлях до вихідного файлу
        title: заголовок ноутбука
        description: опис ноутбука
        imports: список імпортів для додавання
    """
    # Створення ноутбука
    nb = new_notebook()
    
    # Додавання заголовка та опису
    nb.cells.append(new_markdown_cell(f"# {title}\n\n{description}"))
    
    # Додавання імпортів, якщо вони вказані
    if imports:
        imports_code = "\n".join(imports)
        nb.cells.append(new_code_cell(imports_code))
    
    # Збереження ноутбука
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Шаблон ноутбука створено: {output_path}")
    return output_path


def create_evolution_template():
    """Створити шаблон для Evolution_Demo.ipynb"""
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "notebooks", "Evolution_Demo.ipynb")
    
    title = "Навчання агентів для Дилеми в'язня використовуючи Еволюційні Алгоритми"
    
    description = """
Цей ноутбук демонструє використання еволюційних алгоритмів (CMA-ES) для навчання агента 
стратегії в Ітерованій Дилемі В'язня (Iterated Prisoner's Dilemma, IPD).

## Що ми будемо робити:
1. Налаштуємо середовище IPD
2. Реалізуємо еволюційний алгоритм для навчання стратегії
3. Оцінимо продуктивність агента проти різних класичних стратегій
4. Проаналізуємо навчену стратегію та її ефективність

## Очікувані результати:
- Порівняння ефективності еволюційної стратегії з іншими підходами
- Розуміння того, як еволюційні алгоритми можуть знаходити оптимальні стратегії
- Візуалізація процесу навчання та результатів
"""
    
    imports = [
        "import os",
        "import sys",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import pandas as pd",
        "import seaborn as sns",
        "from IPython.display import display",
        "from cmaes import CMA",
        "",
        "# Додаємо шлях до кореневої директорії проекту для можливості імпорту",
        "sys.path.append(os.path.abspath('..'))",
        "",
        "# Імпортуємо наше середовище IPD та стратегії",
        "from env import IPDEnv, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy, PavlovStrategy",
        "",
        "# Налаштування стилю візуалізації",
        "sns.set_style('whitegrid')",
        "plt.rcParams['figure.figsize'] = (12, 6)",
        "plt.rcParams['font.size'] = 12"
    ]
    
    return create_notebook_template(output_path, title, description, imports)


def create_transformer_template():
    """Створити шаблон для Transformer_Demo.ipynb"""
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "notebooks", "Transformer_Demo.ipynb")
    
    title = "Навчання агентів для Дилеми в'язня використовуючи Трансформери"
    
    description = """
Цей ноутбук демонструє використання архітектури трансформерів (Decision Transformer) для навчання агента 
стратегії в Ітерованій Дилемі В'язня (Iterated Prisoner's Dilemma, IPD).

## Що ми будемо робити:
1. Налаштуємо середовище IPD
2. Зберемо дані з успішних партій для навчання трансформера
3. Реалізуємо модель Decision Transformer
4. Оцінимо продуктивність агента проти різних класичних стратегій
5. Проаналізуємо навчену стратегію та її ефективність

## Очікувані результати:
- Порівняння ефективності трансформер-агента з іншими підходами
- Розуміння того, як трансформери можуть навчатися з траєкторій
- Візуалізація процесу навчання та результатів
"""
    
    imports = [
        "import os",
        "import sys",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import pandas as pd",
        "import torch",
        "import torch.nn as nn",
        "import torch.optim as optim",
        "import seaborn as sns",
        "from IPython.display import display",
        "from torch.utils.data import Dataset, DataLoader",
        "",
        "# Додаємо шлях до кореневої директорії проекту для можливості імпорту",
        "sys.path.append(os.path.abspath('..'))",
        "",
        "# Імпортуємо наше середовище IPD та стратегії",
        "from env import IPDEnv, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy, PavlovStrategy",
        "",
        "# Налаштування стилю візуалізації",
        "sns.set_style('whitegrid')",
        "plt.rcParams['figure.figsize'] = (12, 6)",
        "plt.rcParams['font.size'] = 12",
        "",
        "# Перевіряємо доступність GPU",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "print(f'Використовуємо пристрій: {device}')"
    ]
    
    return create_notebook_template(output_path, title, description, imports)


def main():
    parser = argparse.ArgumentParser(description="Утиліти для роботи з Jupyter ноутбуками")
    parser.add_argument("--convert", "-c", metavar="NOTEBOOK", help="Конвертувати ноутбук у Python скрипт")
    parser.add_argument("--clear", "-r", metavar="NOTEBOOK", help="Очистити виведення ноутбука")
    parser.add_argument("--create-evolution", action="store_true", help="Створити шаблон для Evolution_Demo.ipynb")
    parser.add_argument("--create-transformer", action="store_true", help="Створити шаблон для Transformer_Demo.ipynb")
    parser.add_argument("--output-dir", "-o", help="Каталог для виводу (для конвертації)")
    
    args = parser.parse_args()
    
    if args.convert:
        convert_notebook_to_python(args.convert, args.output_dir)
    
    if args.clear:
        clear_notebook_output(args.clear)
    
    if args.create_evolution:
        create_evolution_template()
    
    if args.create_transformer:
        create_transformer_template()
    
    # Якщо не вказано жодного аргументу, виводимо довідку
    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main() 