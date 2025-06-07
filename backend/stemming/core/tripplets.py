from loguru import logger
from typing import List, Dict, Any

from natasha.doc import DocToken
from stemming.core.models import Lang

import re
import logging

# --- Natasha Imports for TripleExtractor ---
try:
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




# --- Triple Extractor Class (User Provided, with refinements) ---
class TripleExtractor:
    def __init__(self):
        if not NATASHA_AVAILABLE:
            raise ImportError("Natasha components are required for TripleExtractor but not installed.")
        # Инициализация компонентов Natasha
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()  # Consider making these optional if Natasha is optional
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)

        # Паттерны для фильтрации стоп-слов при лемматизации извлекаемого текста
        self.lemma_stop_pos = {'PUNCT', 'CCONJ', 'SCONJ', 'PART', 'ADP', 'SYM', 'X', 'INTJ', 'NUM', 'DET'}
        # Паттерны для фильтрации при извлечении самих триплетов (предикатов, например)
        self.triplet_filter_predicates = {'быть', 'являться'}  # Predicates to ignore

    def _preprocess_text_for_natasha(self, text: str) -> str:
        """Предобработка текста для Natasha (внутренний метод)"""
        # Natasha обычно хорошо работает с более-менее сырым текстом,
        # но базовая чистка может помочь.
        # Удаление URL-адресов
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Удаление email-адресов
        text = re.sub(r'\S*@\S*\s?', '', text)
        # Удаление символов, не являющихся буквами, цифрами, пробелами или стандартной пунктуацией,
        # которую Natasha может обработать (дефисы, точки и т.д.).
        # Этот шаг менее агрессивен, чем первоначальный re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'[^\w\s\.\,\-\–\—\:\;\!\?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()  # Нормализация пробелов
        return text  # Natasha сама обрабатывает регистр для теггеров

    def _lemmatize_token_text(self, token_text: str) -> str:
        """Лемматизация отдельного токена/фразы с помощью Natasha"""
        if not token_text or not token_text.strip():
            return ""
        doc = Doc(token_text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        lemmas = []
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
            if token.pos not in self.lemma_stop_pos and len(token.lemma) > 1:  # Filter short lemmas
                lemmas.append(token.lemma)
        return ' '.join(lemmas)

    def extract_spo(self, text: str, language_hint: str) -> List[Dict[str, str]]:
        """Основной метод извлечения SPO триплетов.
           language_hint: 'russian' or 'english'. Currently only 'russian' is supported.
        """
        if language_hint != Lang.Russian.value or not NATASHA_AVAILABLE:
            logger.info(f"Triplet extraction skipped for language '{language_hint}' or Natasha not available.")
            return []

        # Предобработка для Natasha
        prepared_text = self._preprocess_text_for_natasha(text)
        if not prepared_text:
            return []

        doc = Doc(prepared_text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:  # Lemmatize all tokens in doc first
            token.lemmatize(self.morph_vocab)
        doc.parse_syntax(self.syntax_parser)

        triplets = []

        for sent in doc.sents:
            # Находим корневой элемент (сказуемое) - токен с rel 'root'
            # или главный глагол предложения.
            root_token = next((token for token in sent.tokens if token.rel == 'root'), None)

            if not root_token:  # Fallback: find first verb that is head of something
                potential_roots = [t for t in sent.tokens if
                                   t.pos == 'VERB' and any(child.head_id == t.id for child in sent.tokens)]
                if potential_roots:
                    root_token = potential_roots[0]
                else:  # Last fallback: any verb
                    root_token = next((t for t in sent.tokens if t.pos == 'VERB'), None)

            if not root_token or not root_token.lemma:
                continue

            predicate_lemma = root_token.lemma
            if predicate_lemma in self.triplet_filter_predicates or len(predicate_lemma) < 2:
                continue

            # Собираем полный текст предиката, включая вспомогательные слова (AUX, ADVMOD, etc. attached to root)
            # Это более сложная задача, для простоты пока используем только лемму корневого слова.
            # Можно рекурсивно обходить зависимые от root_token слова и собирать фразу.
            # predicate_phrase = self._get_phrase_for_token(root_token, sent.tokens) # Example complex func
            predicate_display = root_token.text  # Use raw text for predicate for better readability in triplet

            subjects = []
            for token in sent.tokens:
                if token.head_id == root_token.id and token.rel == 'nsubj':
                    # Попытка собрать полную именную группу для подлежащего
                    print(f"{token=}")
                    print(f"{sent.tokens=}")
                    subject_phrase_tokens = self._get_noun_phrase_tokens(token, sent.tokens)
                    subject_text = " ".join(t.text for t in subject_phrase_tokens if t.pos not in self.lemma_stop_pos)
                    subject_lemma = " ".join(
                        t.lemma for t in subject_phrase_tokens if t.pos not in self.lemma_stop_pos and len(t.lemma) > 1)

                    if subject_lemma:  # Ensure lemma is not empty after filtering
                        subjects.append({'text': subject_text, 'lemma': subject_lemma})

            objects = []
            for token in sent.tokens:
                if token.head_id == root_token.id and token.rel in {'obj', 'obl', 'iobj'}:
                    object_phrase_tokens = self._get_noun_phrase_tokens(token, sent.tokens)
                    object_text = " ".join(t.text for t in object_phrase_tokens if t.pos not in self.lemma_stop_pos)
                    object_lemma = " ".join(
                        t.lemma for t in object_phrase_tokens if t.pos not in self.lemma_stop_pos and len(t.lemma) > 1)

                    if object_lemma:
                        objects.append({'text': object_text, 'lemma': object_lemma})

            # Формируем триплеты: каждый субъект с каждым объектом (простая комбинаторика)
            # Для более точного связывания нужны более сложные правила или анализ семантических ролей.
            if not subjects and objects:  # Если нет явного субъекта, но есть объект (например, "Построили дом")
                # Используем некий плейсхолдер или пытаемся найти агента из obl:agent
                agent = next((t for t in sent.tokens if t.head_id == root_token.id and t.rel == 'obl:agent'), None)
                if agent:
                    agent_phrase_tokens = self._get_noun_phrase_tokens(agent, sent.tokens)
                    agent_text = " ".join(t.text for t in agent_phrase_tokens if t.pos not in self.lemma_stop_pos)
                    agent_lemma = " ".join(
                        t.lemma for t in agent_phrase_tokens if t.pos not in self.lemma_stop_pos and len(t.lemma) > 1)
                    if agent_lemma:
                        subjects.append({'text': agent_text, 'lemma': agent_lemma})  # Treat agent as subject
                else:
                    subjects.append({'text': 'некто', 'lemma': 'некто'})  # Placeholder subject

            for subj_info in subjects:
                if not objects:  # Если есть субъект, но нет объекта (например, "Кот спит")
                    triplets.append({
                        'subject_text': subj_info['text'],  # Raw text of subject phrase
                        'subject_lemma': subj_info['lemma'],  # Lemmatized subject phrase
                        'predicate_text': predicate_display,  # Raw predicate
                        'predicate_lemma': predicate_lemma,  # Lemmatized predicate
                        'object_text': "",  # No object
                        'object_lemma': "",
                        'sentence': sent.text
                    })
                else:
                    for obj_info in objects:
                        if len(subj_info['lemma']) >= 2 and len(obj_info['lemma']) >= 2:  # Basic filter
                            triplets.append({
                                'subject_text': subj_info['text'],
                                'subject_lemma': subj_info['lemma'],
                                'predicate_text': predicate_display,
                                'predicate_lemma': predicate_lemma,
                                'object_text': obj_info['text'],
                                'object_lemma': obj_info['lemma'],
                                'sentence': sent.text
                            })
        return triplets

    def _get_noun_phrase_tokens(self, head_token: DocToken, all_tokens: List[DocToken]) -> List[DocToken]:
        """
        Собирает все токены, относящиеся к именной группе с head_token во главе.
        Включает сам head_token и его прямые зависимые определители (amod, det),
        а также слова, связанные через nmod, appos, flat.
        Сортирует по позиции в предложении.
        """
        # Use a list instead of a set to avoid hashability issues
        phrase_tokens = [head_token]
        seen_token_ids = {head_token.id}  # Track token IDs to ensure uniqueness

        # Collect dependent tokens forming the phrase
        for token in all_tokens:
            if token.head_id == head_token.id and token.id not in seen_token_ids:
                # Include modifiers and related tokens
                if token.rel in {'amod', 'det', 'nummod', 'nmod', 'appos', 'flat', 'compound'}:
                    phrase_tokens.append(token)
                    seen_token_ids.add(token.id)
                    # Optional: Add recursive call for deeper dependencies (e.g., "очень красивый дом")
                    # phrase_tokens.extend([t for t in self._get_noun_phrase_tokens(token, all_tokens) if t.id not in seen_token_ids])
                    # seen_token_ids.update(t.id for t in self._get_noun_phrase_tokens(token, all_tokens))

        # Sort tokens by their position in the sentence (based on ID, e.g., '1_2' -> 2)
        return sorted(phrase_tokens, key=lambda t: int(t.id.split('_')[-1]))


# --- Updated extract_spo_from_text to use TripleExtractor ---
def actual_extract_spo_from_description(
        description_raw_text: str,  # Pass raw description text
        language_str: str,
        word_id_for_subject: str,
        description_id_for_context: str
) -> List[Dict[str, Any]]:
    """
    Uses the new TripleExtractor to get S-P-O from description's raw text.
    The subject of the triplet in the DB is always the word_id_for_subject.
    The predicate and object come from the extractor.
    """
    if language_str == Lang.Russian.value and russian_triple_extractor:
        # Pass raw description text to TripleExtractor
        extracted_natasha_triplets = russian_triple_extractor.extract_spo(description_raw_text, language_str)
    else:
        logger.info(f"Triplet extraction via Natasha not available or not applicable for lang {language_str}.")
        return []

    db_triplets_data = []
    for nt in extracted_natasha_triplets:
        # nt contains: 'subject_text', 'subject_lemma', 'predicate_text', 'predicate_lemma',
        # 'object_text', 'object_lemma', 'sentence'

        # Per PDF: "Subject триплета - это всегда исходное слово (word)"
        # So, nt['subject_lemma'] and nt['subject_text'] are context from description,
        # but the DB triplet's subject is word_id_for_subject.

        # The predicate and object from Natasha are used as is.
        # Object from Natasha becomes object_literal in our DB triplet.
        # We don't try to map nt['object_lemma'] to an existing Word.id automatically here.

        # Filter out very short or non-meaningful extracted parts
        if not nt['predicate_lemma'] or len(nt['predicate_lemma']) < 2:
            continue
        # Object can be empty if predicate is unary (e.g. "cat sleeps")
        # if not nt['object_lemma'] or len(nt['object_lemma']) < 2:
        #     if nt['object_lemma'] != "": # Allow explicitly empty object for unary
        #         continue

        db_triplets_data.append({
            "subject_id": word_id_for_subject,  # This is the ID of the Word entity
            "subject_type": "word",
            "predicate_raw": nt['predicate_text'],
            "predicate_lemma": nt['predicate_lemma'],
            "object_id": None,  # No auto-linking to other Word entities from this extraction
            "object_type": None,
            "object_literal_raw": nt['object_text'],
            "object_literal_lemma": nt['object_lemma'],
            "language": language_str,
            "source_sentence": nt['sentence'],
            "info": description_id_for_context
        })
    return db_triplets_data

# Global instance of TripleExtractor for Russian
if NATASHA_AVAILABLE:
    russian_triple_extractor = TripleExtractor()
else:
    russian_triple_extractor = None
