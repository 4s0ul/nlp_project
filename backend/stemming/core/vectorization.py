from dataclasses import dataclass

import numpy as np
import logging
import spacy

from enum import Enum
from loguru import logger
from typing import Optional, List
from fastclasses_json import dataclass_json
from sklearn.base import BaseEstimator, TransformerMixin

# --- Natasha Imports for TripleExtractor ---
try:
    from natasha.doc import DocToken
    from natasha import (
        Doc,
        Segmenter,
        MorphVocab,
        NewsEmbedding,
        NewsMorphTagger,
        NewsSyntaxParser
    )

    NATASHA_AVAILABLE = True
except ImportError:
    NATASHA_AVAILABLE = False
    logging.warning("Natasha components not found. Triplet extraction will be disabled. Please install natasha.")


try:
    nlp_en_md = spacy.load("en_core_web_md")
    nlp_ru_md = spacy.load("ru_core_news_md")
except OSError as e:
    logger.error(
        f"Error loading spaCy medium models: {e}. Make sure they are downloaded: python -m spacy download en_core_web_md ru_core_news_md")
    raise



class Lang(Enum):
    Russian = "russian"
    English = "english"


@dataclass_json
@dataclass
class Term:
    term: str
    translation: str
    definition: str
    definition_translated: str
    definition_back_translated: str

    def to_list(self, lang: Lang) -> List[str]:
        if lang == Lang.Russian:
            return [self.translation, self.definition]
        elif lang == Lang.English:
            return [self.term, self.definition_translated]
        raise ValueError(f"Unsupported language: {lang}")


@dataclass_json
@dataclass
class TermVector:
    lang: Optional[Lang] = None
    term: Optional[str] = None
    definition: Optional[str] = None
    term_metadata: Optional[Term] = None

    @staticmethod
    def from_term(term: Term, lang: Lang) -> 'TermVector':
        return TermVector(
            lang=lang,
            term_metadata=term,
            term=term.translation if lang == Lang.Russian else term.term,
            definition=term.definition if lang == Lang.Russian else term.definition_translated
        )


@dataclass_json
@dataclass
class TermVectorized:
    language: Optional[Lang] = None
    term_metadata: Optional[TermVector] = None
    term: Optional[np.ndarray] = None
    definition: Optional[np.ndarray] = None


@dataclass_json
@dataclass
class Data4Graph:
    term: str
    term_vectorized: List[float]
    definition: str
    definition_vector: List[float]

    @staticmethod
    def create_from_term_vectorized(term_vectorized: TermVectorized) -> 'Data4Graph':
        return Data4Graph(
            term=term_vectorized.term_metadata.term,
            term_vectorized=term_vectorized.term.tolist(),
            definition=term_vectorized.term_metadata.definition,
            definition_vector=term_vectorized.definition.tolist()
        )


class AdvancedWord2VecVectorizer(BaseEstimator, TransformerMixin):  # From previous step
    def __init__(self, language_model: spacy.Language):
        self.nlp = language_model

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        vectors = []
        for text in X:
            doc = self.nlp(text)
            if doc.has_vector and doc.vector_norm > 0:
                vectors.append(doc.vector)
            else:
                vectors.append(np.zeros(
                    self.nlp.vocab.vectors_length if self.nlp.vocab.vectors_length > 0 else 1536))  # Fallback dim
        return np.array(vectors)

    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        token = self.nlp.vocab[word]
        if token.has_vector: return token.vector
        doc = self.nlp(word)
        return doc.vector if doc.has_vector and doc.vector_norm > 0 else None



def calculate_similarity(query: str, vectorizer: AdvancedWord2VecVectorizer) -> np.ndarray:
    return vectorizer.transform([query])[0]

def vectorize(term: TermVector, language: Lang, vectorizer: AdvancedWord2VecVectorizer) -> TermVectorized:
    return TermVectorized(
        language=language,
        term_metadata=term,
        term=calculate_similarity(term.term, vectorizer),
        definition=calculate_similarity(term.definition, vectorizer)
    )

vectorizer_ru = AdvancedWord2VecVectorizer(nlp_ru_md)
vectorizer_en = AdvancedWord2VecVectorizer(nlp_en_md)



DEFAULT_VECTOR_DIM = nlp_ru_md.vocab.vectors_length if nlp_ru_md.vocab.vectors_length > 0 else 1536

def get_vectorizer(language: Lang) -> AdvancedWord2VecVectorizer:
    logger.error(f"{language=}")
    if language.value == Lang.Russian.value:
        return vectorizer_ru
    elif language.value == Lang.English.value:
        return vectorizer_en
    raise ValueError(f"Unsupported language for vectorizer: {language}")
