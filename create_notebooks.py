#!/usr/bin/env python3
import nbformat as nbf
import os

# Переконуємося, що директорія notebooks існує
os.makedirs('notebooks', exist_ok=True)

def create_ppo_notebook():
    """Створює демонстраційний ноутбук для PPO-агента"""
    
    nb = nbf.v4.new_notebook()
    
    # Додаємо заголовок і опис
    cells = [
        nbf.v4.new_markdown_cell(
            "# Навчання агентів для Дилеми в'язня використовуючи Proximal Policy Optimization (PPO)\n\n"
            "Цей ноутбук демонструє використання алгоритму глибокого підкріплювального навчання PPO для навчання агента "
            "стратегії в Ітерованій Дилемі В'язня (Iterated Prisoner's Dilemma, IPD).\n\n"
            "## Що ми будемо робити:\n"
            "1. Налаштуємо середовище IPD\n"
            "2. Навчимо PPO-агента проти стратегії Tit-for-Tat\n"
            "3. Оцінимо продуктивність агента проти різних класичних стратегій\n"
            "4. Проаналізуємо навчену стратегію та її ефективність\n\n"
            "## Очікувані результати:\n"
            "- Порівняння ефективності різних стратегій\n"
            "- Розуміння того, як PPO може навчитися кооперативної або конкурентної поведінки\n"
            "- Візуалізація процесу навчання та результатів"
        ),
        
        nbf.v4.new_markdown_cell("## Імпорт необхідних бібліотек"),
        
        nbf.v4.new_code_cell(
            "import os\n"
            "import sys\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import time\n"
            "import seaborn as sns\n"
            "from IPython.display import display\n"
            "from stable_baselines3 import PPO\n"
            "from stable_baselines3.common.monitor import Monitor\n"
            "from stable_baselines3.common.evaluation import evaluate_policy\n\n"
            
            "# Додаємо шлях до кореневої директорії проекту для можливості імпорту\n"
            "sys.path.append(os.path.abspath('..'))\n\n"
            
            "# Імпортуємо наше середовище IPD та стратегії\n"
            "from env import IPDEnv, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy, PavlovStrategy\n"
            "from agents.ppo.train_ppo import create_env, get_cooperation_rates\n\n"
            
            "# Налаштування стилю візуалізації\n"
            "sns.set_style(\"whitegrid\")\n"
            "plt.rcParams['figure.figsize'] = (12, 6)\n"
            "plt.rcParams['font.size'] = 12"
        ),
        
        nbf.v4.new_markdown_cell("## Налаштування параметрів"),
        
        nbf.v4.new_code_cell(
            "# Параметри для створення середовища та навчання\n"
            "NUM_ROUNDS = 10  # Кількість раундів у грі\n"
            "MEMORY_SIZE = 3  # Розмір історії (скільки попередніх ходів пам'ятає агент)\n"
            "SEED = 42       # Початкове значення для відтворюваності результатів\n\n"
            
            "# Шляхи для збереження моделі та результатів\n"
            "MODEL_DIR = \"../models\"\n"
            "RESULTS_DIR = \"../results\"\n\n"
            
            "# Створюємо директорії, якщо вони не існують\n"
            "os.makedirs(MODEL_DIR, exist_ok=True)\n"
            "os.makedirs(RESULTS_DIR, exist_ok=True)\n"
            "os.makedirs(f\"{RESULTS_DIR}/ppo\", exist_ok=True)"
        )
    ]
    
    # Додаємо клітинки для створення середовища
    cells.extend([
        nbf.v4.new_markdown_cell("## Створення та налаштування середовища"),
        
        nbf.v4.new_code_cell(
            "# Створюємо середовище з стратегією Tit-for-Tat як опонентом\n"
            "env = create_env(\n"
            "    opponent_strategy=\"tit_for_tat\",\n"
            "    num_rounds=NUM_ROUNDS,\n"
            "    memory_size=MEMORY_SIZE,\n"
            "    seed=SEED\n"
            ")\n\n"
            
            "# Створюємо окреме середовище для оцінки\n"
            "eval_env = create_env(\n"
            "    opponent_strategy=\"tit_for_tat\",\n"
            "    num_rounds=NUM_ROUNDS,\n"
            "    memory_size=MEMORY_SIZE,\n"
            "    seed=SEED+100  # Інший seed для оцінки\n"
            ")\n\n"
            
            "# Виводимо інформацію про середовище\n"
            "print(f\"Observation space: {env.observation_space}\")\n"
            "print(f\"Action space: {env.action_space}\")\n"
            "print(f\"Number of rounds per game: {NUM_ROUNDS}\")\n"
            "print(f\"Memory size: {MEMORY_SIZE}\")\n"
            "print(f\"Opponent strategy: {env.env.opponent_strategy.name}\")"
        )
    ])
    
    # Додаємо клітинки для навчання агента PPO
    cells.extend([
        nbf.v4.new_markdown_cell(
            "## Навчання PPO-агента\n\n"
            "Тепер ми використаємо алгоритм PPO (Proximal Policy Optimization) з бібліотеки stable-baselines3 для навчання агента. \n\n"
            "PPO - це алгоритм підкріплювального навчання, який оптимізує стратегію (policy) агента ітеративно, обмежуючи "
            "розмір оновлення параметрів між ітераціями для забезпечення стабільності."
        ),
        
        nbf.v4.new_code_cell(
            "# Параметри для навчання PPO\n"
            "learning_rate = 3e-4\n"
            "n_steps = 2048      # Кількість кроків перед кожним оновленням\n"
            "batch_size = 64     # Розмір міні-пакету\n"
            "n_epochs = 10       # Кількість ітерацій навчання на кожному пакеті даних\n"
            "gamma = 0.99        # Коефіцієнт знецінення\n"
            "ent_coef = 0.01     # Коефіцієнт ентропії для заохочення дослідження\n"
            "total_timesteps = 100000  # Зменшено для демонстрації, звичайно використовується 200000-500000\n\n"
            
            "# Ініціалізуємо модель PPO\n"
            "model = PPO(\n"
            "    \"MlpPolicy\",\n"
            "    env,\n"
            "    learning_rate=learning_rate,\n"
            "    n_steps=n_steps,\n"
            "    batch_size=batch_size,\n"
            "    gamma=gamma,\n"
            "    ent_coef=ent_coef,\n"
            "    n_epochs=n_epochs,\n"
            "    verbose=1,\n"
            "    tensorboard_log=f\"{RESULTS_DIR}/ppo_tensorboard\",\n"
            "    seed=SEED\n"
            ")"
        ),
        
        nbf.v4.new_code_cell(
            "# Навчання моделі\n"
            "# Примітка: це може зайняти кілька хвилин\n"
            "start_time = time.time()\n"
            "model.learn(total_timesteps=total_timesteps)\n"
            "training_time = time.time() - start_time\n\n"
            
            "print(f\"Навчання завершено за {training_time:.2f} секунд\")\n\n"
            
            "# Зберігаємо навчену модель\n"
            "model_path = f\"{MODEL_DIR}/ppo_demo.zip\"\n"
            "model.save(model_path)\n"
            "print(f\"Модель збережено у {model_path}\")"
        )
    ])
    
    # Додаємо клітинки для оцінки агента
    cells.extend([
        nbf.v4.new_markdown_cell(
            "## Оцінка навченого агента проти різних опонентів\n\n"
            "Тепер оцінимо, наскільки добре наш агент грає проти різних класичних стратегій."
        ),
        
        nbf.v4.new_code_cell(
            "# Оцінимо модель проти стратегії Tit-for-Tat\n"
            "mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True)\n"
            "print(f\"Середня винагорода проти Tit-for-Tat: {mean_reward:.2f} ± {std_reward:.2f}\")\n\n"
            
            "# Отримуємо рівень кооперації\n"
            "cooperation_rates = get_cooperation_rates(model, eval_env, n_episodes=50)\n"
            "print(f\"Рівень кооперації агента: {cooperation_rates['agent']:.2f}\")\n"
            "print(f\"Рівень кооперації опонента: {cooperation_rates['opponent']:.2f}\")"
        ),
        
        nbf.v4.new_code_cell(
            "# Тепер оцінимо агента проти різних опонентів\n"
            "opponent_strategies = {\n"
            "    \"tit_for_tat\": TitForTat(),\n"
            "    \"always_cooperate\": AlwaysCooperate(),\n"
            "    \"always_defect\": AlwaysDefect(),\n"
            "    \"random\": RandomStrategy(seed=SEED+200),\n"
            "    \"pavlov\": PavlovStrategy()\n"
            "}\n\n"
            
            "# Проводимо оцінку\n"
            "results = []\n\n"
            
            "for opponent_name, opponent_strategy in opponent_strategies.items():\n"
            "    # Створюємо середовище з поточним опонентом\n"
            "    env = create_env(\n"
            "        opponent_strategy=opponent_strategy,\n"
            "        num_rounds=NUM_ROUNDS,\n"
            "        memory_size=MEMORY_SIZE,\n"
            "        seed=SEED+300\n"
            "    )\n"
            "    \n"
            "    # Оцінюємо модель\n"
            "    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)\n"
            "    \n"
            "    # Отримуємо рівні кооперації\n"
            "    cooperation_rates = get_cooperation_rates(model, env, n_episodes=50)\n"
            "    \n"
            "    # Зберігаємо результати\n"
            "    results.append({\n"
            "        \"opponent\": opponent_name,\n"
            "        \"mean_reward\": mean_reward,\n"
            "        \"std_reward\": std_reward,\n"
            "        \"agent_cooperation_rate\": cooperation_rates[\"agent\"],\n"
            "        \"opponent_cooperation_rate\": cooperation_rates[\"opponent\"]\n"
            "    })\n\n"
            
            "# Створюємо DataFrame з результатами\n"
            "results_df = pd.DataFrame(results)\n"
            "display(results_df)"
        )
    ])
    
    # Додаємо клітинки для візуалізації результатів
    cells.extend([
        nbf.v4.new_markdown_cell("## Візуалізація результатів"),
        
        nbf.v4.new_code_cell(
            "# Візуалізуємо середню винагороду для різних опонентів\n"
            "plt.figure(figsize=(12, 6))\n"
            "sns.barplot(x=\"opponent\", y=\"mean_reward\", data=results_df, palette=\"viridis\")\n"
            "plt.title(\"Середня винагорода PPO-агента проти різних опонентів\")\n"
            "plt.xlabel(\"Опонент\")\n"
            "plt.ylabel(\"Середня винагорода\")\n"
            "plt.grid(True, alpha=0.3)\n"
            "plt.show()"
        ),
        
        nbf.v4.new_code_cell(
            "# Візуалізуємо рівні кооперації\n"
            "plt.figure(figsize=(12, 6))\n\n"
            
            "# Підготовка даних для групового стовпчикового графіка\n"
            "x = np.arange(len(results_df))\n"
            "width = 0.35\n\n"
            
            "plt.bar(x - width/2, results_df[\"agent_cooperation_rate\"], width, label=\"PPO Agent\")\n"
            "plt.bar(x + width/2, results_df[\"opponent_cooperation_rate\"], width, label=\"Opponent\")\n\n"
            
            "plt.title(\"Рівні кооперації PPO-агента та опонентів\")\n"
            "plt.xlabel(\"Опонент\")\n"
            "plt.ylabel(\"Рівень кооперації\")\n"
            "plt.xticks(x, results_df[\"opponent\"])\n"
            "plt.ylim(0, 1.05)\n"
            "plt.legend()\n"
            "plt.grid(True, alpha=0.3)\n"
            "plt.show()"
        )
    ])
    
    # Додаємо клітинки для аналізу окремої гри та висновків
    cells.extend([
        nbf.v4.new_markdown_cell(
            "## Перегляд ходу окремої гри\n\n"
            "Подивимося на один конкретний матч між PPO-агентом та стратегією Tit-for-Tat."
        ),
        
        nbf.v4.new_code_cell(
            "# Створюємо середовище для демонстрації\n"
            "demo_env = create_env(\n"
            "    opponent_strategy=\"tit_for_tat\",\n"
            "    num_rounds=NUM_ROUNDS,\n"
            "    memory_size=MEMORY_SIZE,\n"
            "    seed=SEED\n"
            ")\n\n"
            
            "# Скидаємо середовище\n"
            "observation, _ = demo_env.reset()\n"
            "done = False\n"
            "total_reward = 0\n"
            "game_history = []\n\n"
            
            "# Проходимо одну гру\n"
            "while not done:\n"
            "    # Отримуємо дію від агента\n"
            "    action, _ = model.predict(observation, deterministic=True)\n"
            "    \n"
            "    # Виконуємо крок у середовищі\n"
            "    observation, reward, terminated, truncated, info = demo_env.step(action)\n"
            "    done = terminated or truncated\n"
            "    \n"
            "    # Зберігаємо інформацію\n"
            "    total_reward += reward\n"
            "    game_history.append(info)\n\n"
            
            "# Виводимо інформацію про гру\n"
            "print(f\"Загальна винагорода: {total_reward}\")\n"
            "print(f\"Рахунок: Агент {demo_env.env.player_score} - {demo_env.env.opponent_score} Опонент\")\n\n"
            
            "# Створюємо DataFrame для візуалізації історії гри\n"
            "history_df = pd.DataFrame(game_history)\n"
            "history_df[\"action_name\"] = history_df[\"player_action\"].apply(lambda x: \"Cooperate\" if x == 0 else \"Defect\")\n"
            "history_df[\"opponent_action_name\"] = history_df[\"opponent_action\"].apply(lambda x: \"Cooperate\" if x == 0 else \"Defect\")\n"
            "display(history_df[[\"round\", \"action_name\", \"opponent_action_name\", \"player_payoff\", \"opponent_payoff\"]])"
        ),
        
        nbf.v4.new_code_cell(
            "# Візуалізуємо кооперацію та винагороди протягом гри\n"
            "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)\n\n"
            
            "# Графік дій (кооперація/зрада)\n"
            "ax1.plot(history_df[\"round\"], history_df[\"player_action\"], 'o-', label=\"PPO Agent\")\n"
            "ax1.plot(history_df[\"round\"], history_df[\"opponent_action\"], 'o--', label=\"Tit-for-Tat\")\n"
            "ax1.set_ylabel(\"Дія (0=Кооперація, 1=Зрада)\")\n"
            "ax1.set_title(\"Дії протягом гри\")\n"
            "ax1.legend()\n"
            "ax1.grid(True, alpha=0.3)\n"
            "ax1.set_yticks([0, 1])\n"
            "ax1.set_yticklabels([\"Cooperate\", \"Defect\"])\n\n"
            
            "# Графік винагород\n"
            "ax2.plot(history_df[\"round\"], history_df[\"player_payoff\"], 'o-', label=\"PPO Agent\")\n"
            "ax2.plot(history_df[\"round\"], history_df[\"opponent_payoff\"], 'o--', label=\"Tit-for-Tat\")\n"
            "ax2.set_xlabel(\"Раунд\")\n"
            "ax2.set_ylabel(\"Винагорода\")\n"
            "ax2.set_title(\"Винагороди протягом гри\")\n"
            "ax2.legend()\n"
            "ax2.grid(True, alpha=0.3)\n\n"
            
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        
        nbf.v4.new_markdown_cell(
            "## Висновки\n\n"
            "Ми успішно навчили PPO-агента грати в Ітеровану Дилему В'язня. Основні спостереження:\n\n"
            "1. PPO може навчитися ефективній стратегії для гри з різними опонентами\n"
            "2. Агент може адаптувати свою поведінку залежно від опонента, з яким він взаємодіє\n"
            "3. Рівень кооперації агента залежить від того, проти якого опонента він був навчений\n\n"
            
            "Для поліпшення результатів можна спробувати:\n"
            "- Тренувати проти різних опонентів або міксу стратегій\n"
            "- Збільшити кількість раундів у грі\n"
            "- Експериментувати з параметрами PPO\n"
            "- Змінити структуру винагороди для заохочення певних типів поведінки\n\n"
            
            "PPO є потужним алгоритмом для вирішення задач Дилеми В'язня, особливо коли потрібна адаптивна стратегія, "
            "яка може змінюватись залежно від поведінки опонента."
        )
    ])
    
    nb['cells'] = cells
    
    # Задаємо метадані для ноутбука
    nb.metadata = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {
                'name': 'ipython',
                'version': 3
            },
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.9.13'
        }
    }
    
    # Зберігаємо ноутбук у файл
    with open('notebooks/PPO_Demo.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Ноутбук 'notebooks/PPO_Demo.ipynb' успішно створено")


if __name__ == "__main__":
    create_ppo_notebook() 