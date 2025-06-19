
from collections import Counter
import matplotlib.pyplot as plt
from stemming.core.preprocessing import TextPreprocessor, STEMMERS, nlp_ru, nlp_en

class StemmingEvaluator:
    @staticmethod
    def analyze_stemming_quality(text: str, language: str, sample_size=20, top_n=10):
        stemmer = STEMMERS[language]
        cleaned_text, words = TextPreprocessor.preprocess(text, language).split()
        stems = [stemmer.stem(w) for w in words]

        stats = {
            "total_words": len(words),
            "unique_stems": len(set(stems)),
            "stem_length_changes": sum(len(s) != len(w) for s, w in zip(stems, words)),
            "most_common_stems": Counter(stems).most_common(top_n)
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.axis('off')
        ax1.set_title("Примеры стемминга")
        ax1.table(cellText=list(zip(words, stems))[:sample_size], colLabels=["Оригинал", "Стем"], loc="center")
        ax2.hist([len(s) for s in stems], bins=range(1, max(map(len, stems)) + 1))
        ax2.set_title("Распределение длин основ")
        ax2.set_xlabel("Длина")
        ax2.set_ylabel("Количество")

        return stats, fig
