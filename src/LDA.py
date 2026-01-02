#LDA_на_токенах

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import nltk
import spacy
import matplotlib.cm as cm
import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from nltk.corpus import stopwords

# nltk.download('punkt')
# nltk.download('stopwords')

def read_texts(folder):
    texts, filenames = [], []
    for file_path in Path(folder).glob('*.txt'):
        with file_path.open('r', encoding='utf-8') as f:
            texts.append(f.read())
        filenames.append(file_path.name)
    return texts, filenames

def preprocess(text, nlp, all_stopwords):
    doc = nlp(text.lower())
    return [
        token.lemma_ for token in doc
        if token.is_alpha
        and token.lemma_ not in all_stopwords
        and len(token) > 3
        and token.pos_ in {'NOUN', 'ADJ', 'VERB'}
    ]


def plot_lda_topics(lda_model, num_words=15):
    num_topics = lda_model.num_topics

    if num_topics == 4:
        nrows, ncols = 2, 2
    else:
        ncols = 2
        nrows = (num_topics + 1) // 2

    topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
    axes = axes.flatten()



    for i, (topic_id, topic) in enumerate(topics):
        words, weights = zip(*topic)
        axes[i].barh(words, weights, color='#4B8BBE')
        axes[i].invert_yaxis()
        axes[i].set_title(f"Topic {topic_id + 1}", fontsize=10)
        axes[i].tick_params(axis='x', labelsize=8)
        axes[i].tick_params(axis='y', labelsize=8)

        # Выводим слова в консоль
        print(f"\nTopic {topic_id + 1}:")
        for w, p in topic:
            print(f"  {w:<20} {p:.4f}")

    # Скрываем лишние оси, если они есть
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def get_topic_distribution(texts, dictionary, lda_model):
    dists = []
    for text in texts:
        bow = dictionary.doc2bow(text)
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0.0)
        topic_dist = [prob for _, prob in sorted(topic_dist)]
        dists.append(topic_dist)
    return dists

def main():
    CHILDREN_STALINIST_PLAYS_FOLDER = '../plays_lemm/Childrens_Stalinist_Plays'
    CHILDREN_THAW_PLAYS_FOLDER = '../plays_lemm/Childrens_Thaw_Plays'
    METADATA_PATH = '../dataframe.csv'

    # --- Загрузка spaCy и стоп-слов ---
    nlp = spacy.load("ru_core_news_sm")
    stop_words = set(stopwords.words('russian'))
    custom_stopwords = {
        "это", "весь", "ещё", "сказать", "знать", "говорить",
        "мочь", "хотеть", "пойти", "идти", "давать", "быть", "нету",
        "наш", "ваш", "свой", "ничто", "тот", "такой", "здравствовать",
        "он", "она", "они", "мы", "вы", "там", "тут", "самый"
    }
    all_stopwords = stop_words.union(custom_stopwords)

    # --- Чтение текстов ---
    stalinist_texts, stalinist_filenames = read_texts(CHILDREN_STALINIST_PLAYS_FOLDER)
    thaw_texts, thaw_filenames = read_texts(CHILDREN_THAW_PLAYS_FOLDER)

    # --- Обработка текстов ---
    stalinist_processed = [preprocess(doc, nlp, all_stopwords) for doc in stalinist_texts]
    thaw_processed = [preprocess(doc, nlp, all_stopwords) for doc in thaw_texts]

    all_texts = stalinist_processed + thaw_processed
    all_filenames = [f.replace('.txt','').strip().lower() for f in stalinist_filenames + thaw_filenames]

    # Преобразуем токены в строки для gensim
    all_texts_str = [[str(token) for token in text] for text in all_texts]

    # --- Чтение метаданных ---
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    metadata = pd.read_csv(METADATA_PATH, encoding='utf-8')
    metadata['filename'] = metadata['filename'].astype(str).str.strip().str.lower()
    metadata['year'] = pd.to_numeric(metadata['year'], errors='coerce')
    metadata = metadata.dropna(subset=['year'])
    metadata['decade'] = (metadata['year'] // 5) * 5

    # # --- Создание DataFrame с текстами и объединение с метаданными ---
    df = pd.DataFrame({'filename': all_filenames, 'tokens': all_texts})
    merged = pd.merge(df, metadata, on='filename', how='inner')
    print(f"Совпало файлов: {len(merged)} из {len(df)}")

    print("\n=== Проверка распределения пьес по десятилетиям ===")
    if 'decade' in merged.columns:
        decade_groups = merged.groupby('decade')['filename'].apply(list)
        for decade, plays in sorted(decade_groups.items()):
            print(f"\n{decade}-е годы ({len(plays)} пьес):")
            for p in plays:
                print(f"  - {p}")
    else:
        print("Столбец 'decade' не найден!")

    if merged.empty:
        raise ValueError("Не найдено совпадений между файлами и метаданными!")

    # --- Подготовка для LDA ---
    dictionary = corpora.Dictionary(all_texts_str)
    dictionary.filter_extremes(no_below=3, no_above=0.4)
    corpus = [dictionary.doc2bow(text) for text in all_texts_str]

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=4,
        passes=25,
        iterations=500,
        alpha='auto',
        eta='auto',
        random_state=68,
        per_word_topics=True
    )

    num_topics = lda_model.num_topics
    colors = cm.get_cmap('tab10', num_topics).colors
    topic_names = [f"Topic {i + 1}" for i in range(num_topics)]
    coherence_model_lda = CoherenceModel(model=lda_model, texts=all_texts_str, dictionary=dictionary, coherence='c_v')
    print("Coherence Score:", coherence_model_lda.get_coherence())

    # --- График топиков ---
    plot_lda_topics(lda_model, num_words=15)

    # --- Распределение по эрам (по папкам) ---
    stalin_texts_tokens = [[str(token) for token in text] for text in stalinist_processed]
    thaw_texts_tokens = [[str(token) for token in text] for text in thaw_processed]

    stalinskie_dists = get_topic_distribution(stalin_texts_tokens, dictionary, lda_model)
    ottepel_dists = get_topic_distribution(thaw_texts_tokens, dictionary, lda_model)

    avg_stalin = np.mean(stalinskie_dists, axis=0)
    avg_ottepel = np.mean(ottepel_dists, axis=0)

    x = np.arange(num_topics)
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, avg_stalin, width=width, label='Late Stalinist', color='#4B8BBE')  # яркий синий
    plt.bar(x + width / 2, avg_ottepel, width=width, label='Thaw', color='#FFA500')  # контрастный оранжевый
    plt.xlabel('Topic number')
    plt.ylabel('Average probability')
    plt.title('Distribution of topics by era')
    plt.xticks(x, [f'Topic {i + 1}' for i in range(num_topics)])
    plt.legend()
    plt.tight_layout()
    plt.show()


    decade_topic_distribution = {}
    for decade, group in merged.groupby('decade'):
        texts = group['tokens'].tolist()
        dists = get_topic_distribution(texts, dictionary, lda_model)
        if len(dists) > 0:
            avg = np.mean(dists, axis=0)
            decade_topic_distribution[decade] = avg
        else:
            decade_topic_distribution[decade] = np.zeros(lda_model.num_topics)

    # --- Распределение по десятилетиям ---
    decades = sorted(merged['decade'].unique())
    avg_by_decade = []

    for d in decades:
        decade_texts = merged.loc[merged['decade'] == d, 'tokens'].tolist()
        if decade_texts:
            decade_texts_tokens = [[str(token) for token in text] for text in decade_texts]
            dists = get_topic_distribution(decade_texts_tokens, dictionary, lda_model)
            avg_by_decade.append(np.mean(dists, axis=0))
        else:
            avg_by_decade.append(np.zeros(lda_model.num_topics))

    avg_by_decade = np.array(avg_by_decade)

    # --- Столбчатая диаграмма по десятилетиям ---
    colors = cm.get_cmap('tab10', num_topics).colors

    decades = sorted(decade_topic_distribution.keys())
    x = np.arange(len(decades))
    width = 0.2


    plt.figure(figsize=(10, 6))
    for i in range(num_topics):
        values = [decade_topic_distribution[d][i] for d in decades]
        plt.bar(x + i * width, values, width=width, color=colors[i], label=f'Topic {i + 1}')

    plt.xlabel('Decade')
    plt.ylabel('Average topic proportion')
    plt.title('Distribution of topics by five-year periods')
    plt.xticks(x + width * (num_topics - 1) / 2, decades)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


    stalinist_df = pd.DataFrame({
        'filename': [f.replace('.txt', '') for f in stalinist_filenames],
        'tokens': stalinist_processed,
        'period': 'Stalinist'
    })

    thaw_df = pd.DataFrame({
        'filename': [f.replace('.txt', '') for f in thaw_filenames],
        'tokens': thaw_processed,
        'period': 'Thaw'
    })

    # Объединяем в один общий DataFrame
    df = pd.concat([stalinist_df, thaw_df], ignore_index=True)
    # Добавим имена топиков (для удобства)
    topic_names = [f"Topic {i + 1}" for i in range(lda_model.num_topics)]

    # Создаём DataFrame с распределением тем по пьесам
    topic_dists = get_topic_distribution(all_texts, dictionary, lda_model)
    topic_df = pd.DataFrame(topic_dists, columns=topic_names)
    topic_df['filename'] = merged['filename']
    topic_df['period'] = merged['period']
    topic_df['year'] = merged['year']

    # Разделяем на эпохи
    stalin_df = topic_df[topic_df['period'] == '1945—1953'].sort_values(by='year')
    thaw_df = topic_df[topic_df['period'] == '1954—1964'].sort_values(by='year')

    # Цвета — совпадают с графиками по десятилетиям
    colors = cm.get_cmap('tab10', len(topic_names)).colors

    def plot_stacked_topics(data, title):
        if data.empty:
            print(f"Нет данных для {title}")
            return

        fig, ax = plt.subplots(figsize=(10, len(data) * 0.25))

        bottom = np.zeros(len(data))
        for i, topic in enumerate(topic_names):
            ax.barh(data['filename'], data[topic], left=bottom, color=colors[i], label=topic)
            bottom += data[topic]

        ax.set_xlabel('Topic proportion')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    # Отдельные графики по эпохам
    plot_stacked_topics(stalin_df, "Topic Composition per Play — Late Stalinist")
    plot_stacked_topics(thaw_df, "Topic Composition per Play — Thaw")



if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
