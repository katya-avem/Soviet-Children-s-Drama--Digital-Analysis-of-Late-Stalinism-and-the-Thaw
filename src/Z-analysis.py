import os
from collections import Counter
import math
from tqdm import tqdm

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

russian_stopwords = set(stopwords.words("russian"))
custom_stopwords = {"это", "тот", "такой", "весь", "сам", "ещё"}
all_stopwords = russian_stopwords | custom_stopwords

def read_words_from_folder(folder_path):
    all_words = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), encoding="utf-8") as f:
                text = f.read().lower()
                words = text.split()
                all_words.extend(words)
    return all_words

def compute_z_score(f1, f2, n1, n2):
    p = (f1 + f2) / (n1 + n2)
    sigma = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (f1 / n1 - f2 / n2) / sigma if sigma != 0 else 0

# Путь к вашим данным
path_stalin = "../plays_lemm/Childrens_Stalinist_Plays"
path_ottepel = "../plays_lemm/Childrens_Thaw_Plays"

# Читаем слова
print("Читаем корпус Сталина...")
words_stalin = read_words_from_folder(path_stalin)
print("Читаем корпус Оттепели...")
words_ottepel = read_words_from_folder(path_ottepel)

# Подсчёт частот
counter_stalin = Counter(words_stalin)
counter_ottepel = Counter(words_ottepel)

n1 = len(words_stalin)
n2 = len(words_ottepel)

# Все уникальные слова
all_words = {w for w in counter_stalin | counter_ottepel if w not in all_stopwords}

# Z-анализ
results = []
for word in tqdm(all_words, desc="Calculating Z-scores"):
    if word in all_stopwords:
        continue
    f1 = counter_stalin.get(word, 0)
    f2 = counter_ottepel.get(word, 0)
    z = compute_z_score(f1, f2, n1, n2)
    results.append((word, f1, f2, z))

# Сортировка по модулю Z
results.sort(key=lambda x: abs(x[3]), reverse=True)

# Сохраняем в файл
with open("z_analysis_results.tsv", "w", encoding="utf-8") as out:
    out.write("word\tfreq_stalin\tfreq_ottepel\tz_score\n")
    for word, f1, f2, z in results:
        out.write(f"{word}\t{f1}\t{f2}\t{z:.3f}\n")

import pandas as pd
import matplotlib.pyplot as plt

# Загружаем данные
df = pd.read_csv("z_analysis_results.tsv", sep="\t")

# Отбрасываем редкие слова (опционально)
df = df[(df['freq_stalin'] + df['freq_ottepel']) >= 5]

# Разделяем слова по эпохам
top_stalin = df[df['z_score'] > 0].sort_values(by="z_score", ascending=False).head(30)
top_ottepel = df[df['z_score'] < 0].sort_values(by="z_score").head(30)

# Функция построения графика
def plot_z_words(df, title, color):
    plt.figure(figsize=(8, 10))
    plt.barh(df['word'], df['z_score'], color=color)
    plt.xlabel(""
               "")
    plt.title(title)
    plt.gca().invert_yaxis()  # Чтобы топ был сверху
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Визуализация
plot_z_words(top_stalin, "keywords for children's plays of the late Stalinist period", color="darkred")
plot_z_words(top_ottepel, "keywords for children's plays of the Thaw period", color="navy")