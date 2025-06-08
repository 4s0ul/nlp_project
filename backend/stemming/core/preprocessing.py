
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
import spacy
from spacy.tokens import Token

from stemming.core.coreferences import resolve_coreference_with_natasha_rules, resolve_coreferences_english_with_spacy
from stemming.core.models import Lang

STOP_WORDS = {
    "english": set(stopwords.words("english")),
    "russian": set(stopwords.words("russian"))
}

TEXT_CLEANING_PATTERN = {
    "english": re.compile(r"[^a-zA-Zа-яА-Я0-9 ]"),  # Note the space after 9
    "russian": re.compile(r"[^а-яА-Яa-zA-Z0-9 ]")
}

def clean_text(text: str, lang: str = "english") -> str:
    # Step 1: Remove unwanted characters (preserve letters, digits, and spaces)
    text = TEXT_CLEANING_PATTERN[lang].sub("", text)
    # Step 2: Replace multiple spaces with a single space
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

STEMMERS = {
    "english": PorterStemmer(),
    "russian": SnowballStemmer("russian")
}

try:
    nlp_en_sm = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp_ru_sm = spacy.load("ru_core_news_sm", disable=["parser", "ner"])  # Used for non-Natasha preprocessing
except OSError as e:
    logger.error(
        f"Error loading spaCy small models: {e}. Make sure they are downloaded: python -m spacy download en_core_web_sm ru_core_news_sm")
    raise


try:
    nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp_ru = spacy.load("ru_core_news_sm", disable=["parser", "ner"])
    nlp_en_trf = None 
    # nlp_en_trf = spacy.load("en_coreference_web_trf")
except OSError:
    raise RuntimeError("spaCy models not found. Install them via 'python -m spacy download ...'")


class TextPreprocessor:  # This is the general preprocessor from the previous step
    @staticmethod
    def preprocess(text: str, language: str) -> str:
        # Coreference resolution (user requested to keep it as a placeholder)
        # if language == Lang.Russian.value:
        #     text = resolve_coreference_with_natasha_rules(text, language)

        text_lower = text.lower()
        # Clean numbers and special characters but keep spaces for word splitting
        # text_cleaned = TEXT_CLEANING_PATTERN[language].sub("", text_lower)
        text_cleaned = clean_text(text=text_lower, lang=language)
        words = text_cleaned.split()
        words = [w for w in words if w not in STOP_WORDS[language] and len(w) > 1]

        stemmer_to_use = STEMMERS.get(language)
        if stemmer_to_use:
            stemmed_words = [stemmer_to_use.stem(word) for word in words]
        else:  # Fallback if no stemmer for the language
            stemmed_words = words

        final_tokens = []
        # Use the appropriate spaCy model for tokenizing the *stemmed* words for filtering
        # This step seems unusual (spaCy on stemmed words), but kept from original user code structure
        tokenizer_for_filter = None
        if language == Lang.English.value:
            tokenizer_for_filter = nlp_en_sm
        elif language == Lang.Russian.value:
            tokenizer_for_filter = nlp_ru_sm

        if tokenizer_for_filter:
            doc_for_spacy_filter = tokenizer_for_filter(" ".join(stemmed_words))
            for token in doc_for_spacy_filter:
                if not token.is_stop and not token.is_punct and not token.is_space and len(token.text) > 1:
                    final_tokens.append(token.text)
        else:  # If no tokenizer, use stemmed words directly after NLTK stopword removal
            final_tokens = stemmed_words

        return " ".join(final_tokens)


