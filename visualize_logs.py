import json
import matplotlib.pyplot as plt

# Проверка доступности Seaborn
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Загрузка данных из файла
def load_logs(log_file):
    with open(log_file, "r") as f:
        logs = json.load(f)
    return logs

# Визуализация логов с помощью Matplotlib
def visualize_with_matplotlib(logs):
    episode_rewards = logs["episode_rewards"]
    episode_lengths = logs["episode_lengths"]

    # График наград
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label="Награда за эпизод")
    plt.xlabel("Эпизод")
    plt.ylabel("Награда")
    plt.title("Награды за эпизоды")
    plt.legend()
    plt.grid()
    plt.show()

    # График длительности эпизодов
    plt.figure(figsize=(12, 6))
    plt.plot(episode_lengths, label="Длительность эпизода")
    plt.xlabel("Эпизод")
    plt.ylabel("Длительность")
    plt.title("Длительность эпизодов")
    plt.legend()
    plt.grid()
    plt.show()

# Визуализация логов с помощью Seaborn
def visualize_with_seaborn(logs):
    if not SEABORN_AVAILABLE:
        print("Seaborn не установлен. Используйте Matplotlib для визуализации.")
        visualize_with_matplotlib(logs)
        return

    episode_rewards = logs["episode_rewards"]
    episode_lengths = logs["episode_lengths"]

    # Настройка стиля Seaborn
    sns.set(style="darkgrid")

    # График наград
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=range(len(episode_rewards)), y=episode_rewards, label="Награда за эпизод")
    plt.xlabel("Эпизод")
    plt.ylabel("Награда")
    plt.title("Награды за эпизоды")
    plt.legend()
    plt.show()

    # График длительности эпизодов
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=range(len(episode_lengths)), y=episode_lengths, label="Длительность эпизода")
    plt.xlabel("Эпизод")
    plt.ylabel("Длительность")
    plt.title("Длительность эпизодов")
    plt.legend()
    plt.show()

# Основная функция
def main():
    log_file = "/Users/dasha/Documents/adversarial_framework/training_log.json" # Путь к файлу с логами
    print(log_file)
    logs = load_logs(log_file)

    # Выбор визуализации
    print("Выберите способ визуализации:")
    print("1. Matplotlib")
    if SEABORN_AVAILABLE:
        print("2. Seaborn")
    choice = input("Введите номер (1 или 2): ")

    if choice == "1":
        visualize_with_matplotlib(logs)
    elif choice == "2" and SEABORN_AVAILABLE:
        visualize_with_seaborn(logs)
    else:
        print("Неверный выбор или Seaborn не установлен. Используем Matplotlib.")
        visualize_with_matplotlib(logs)

if __name__ == "__main__":
    main()