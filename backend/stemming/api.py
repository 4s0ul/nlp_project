# Standard library imports
import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any, Generator, Tuple

# Third-party library imports
import numpy as np
import networkx as nx
import nltk
import spacy
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from natasha.doc import DocToken
from natasha import (
    Doc,
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser
)
from pgvector.sqlalchemy import Vector
from sklearn.base import BaseEstimator, TransformerMixin
from sqlmodel import Field, SQLModel, create_engine, Session, select, delete, Column, or_, Relationship
from sqlalchemy import text as sa_text, TEXT

# Local application/library imports
from stemming.core.graph import build_knowledge_graph, visualize_knowledge_graph
from stemming.core.models import (
    Lang,
    create_db_and_tables,
    TopicResponse,
    get_session,
    generate_sha256_id,
    Topic,
    Word,
    PreprocessResponse,
    TextProcessRequest,
    VectorizeResponse,
    WordResponse,
    WordCreateRequest,
    Description,
    Embedding,
    Triplet,
    DescriptionResponse,
    TripletResponse,
    TripletCreateRequest,
    GraphNode,
    GraphEdge
)
from stemming.core.preprocessing import TextPreprocessor
from stemming.core.tripplets import russian_triple_extractor, actual_extract_spo_from_description
from stemming.core.vectorization import get_vectorizer, DEFAULT_VECTOR_DIM

# Check for Natasha availability
NATASHA_AVAILABLE = True
try:
    from natasha import (
        Doc,
        Segmenter,
        MorphVocab,
        NewsEmbedding,
        NewsMorphTagger,
        NewsSyntaxParser
    )
except ImportError:
    NATASHA_AVAILABLE = False
    logging.warning("Natasha components not found. Triplet extraction will be disabled. Please install natasha.")

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Словарь API", version="1.1")  # Version bump
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or "*" for testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        logger.info("NLTK stopwords not found. Downloading...")
        nltk.download('stopwords')
    except Exception as e:
        logger.warning(f"Could not verify NLTK stopwords: {e}. Download might be needed.")

    # Uncomment if you want tables to be created on startup (useful for dev/testing)
    # In production, migrations (e.g. Alembic) are preferred.
    create_db_and_tables()
    logger.info("Application startup: NLTK stopwords checked. DB tables ensured (created if not exist).")
    if not NATASHA_AVAILABLE:
        logger.warning("Natasha library not found. Russian triplet extraction will be disabled.")
    else:
        logger.info("Natasha library found. Russian triplet extraction is available.")


class TopicCreateRequest(SQLModel): name: str; info: Optional[str] = None


def get_lang_str(language: Lang) -> str: return language.value


# --- Topics API (Largely unchanged) ---
@app.post("/topics/", response_model=TopicResponse, tags=["Topics"])
def create_topic_endpoint(topic_data: TopicCreateRequest, session: Session = Depends(get_session)):
    topic_id = generate_sha256_id(topic_data.name.lower().strip())
    if session.get(Topic, topic_id):
        raise HTTPException(status_code=409, detail=f"Topic '{topic_data.name}' already exists.")
    db_topic = Topic(id=topic_id, name=topic_data.name.strip(), info=topic_data.info)
    session.add(db_topic);
    session.commit();
    session.refresh(db_topic)
    return db_topic


@app.get("/topics/", response_model=List[TopicResponse], tags=["Topics"])
def get_topics_endpoint(session: Session = Depends(get_session)): return session.exec(select(Topic)).all()


@app.get("/topics/{topic_id}", response_model=TopicResponse, tags=["Topics"])
def get_topic_endpoint(topic_id: str, session: Session = Depends(get_session)):
    topic = session.get(Topic, topic_id)
    if not topic: raise HTTPException(status_code=404, detail="Topic not found")
    return topic


@app.put("/topics/{topic_id}", response_model=TopicResponse, tags=["Topics"])
def update_topic_endpoint(topic_id: str, topic_data: TopicCreateRequest, session: Session = Depends(get_session)):
    db_topic = session.get(Topic, topic_id)
    if not db_topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    new_name_stripped = topic_data.name.strip()
    new_id = generate_sha256_id(new_name_stripped.lower())

    if new_id != topic_id:
        # Create new topic and update related words
        new_topic = Topic(id=new_id, name=new_name_stripped, info=topic_data.info,
                          created_at=datetime.now(timezone.utc))
        session.add(new_topic)

        # Update words to reference new topic
        words = session.exec(select(Word).where(Word.topic_id == topic_id)).all()
        for word in words:
            word.topic_id = new_id
            session.add(word)

        # Delete old topic
        session.delete(db_topic)
    else:
        db_topic.name = new_name_stripped
        db_topic.info = topic_data.info
        session.add(db_topic)

    session.commit()
    session.refresh(db_topic if new_id == topic_id else new_topic)
    return db_topic if new_id == topic_id else new_topic


@app.delete("/topics/{topic_id}", tags=["Topics"])
def delete_topic_endpoint(topic_id: str, cascade: bool = Query(False, description="Cascade delete related items"),
                          session: Session = Depends(get_session)):
    db_topic = session.get(Topic, topic_id)
    if not db_topic: raise HTTPException(status_code=404, detail="Topic not found")
    if cascade:
        words_in_topic = session.exec(select(Word).where(Word.topic_id == topic_id)).all()
        for word_obj in words_in_topic:  # Call delete_word_endpoint's logic for each word
            _delete_word_data(word_obj.id, session, commit_changes=False)  # Internal helper
        session.commit()  # Commit all deletions from words
    remaining_words = session.exec(select(Word).where(Word.topic_id == topic_id).limit(1)).first()
    if remaining_words:
        raise HTTPException(status_code=400, detail=f"Topic has words. Delete them or use cascade=true.")
    session.delete(db_topic);
    session.commit()
    return {"ok": True, "message": f"Topic '{db_topic.name}' and related items (if cascade) deleted."}


# --- Processing API (Largely unchanged) ---
@app.post("/preprocess/", response_model=PreprocessResponse, tags=["Processing"])
async def preprocess_text_endpoint(request: TextProcessRequest):
    lang_str = get_lang_str(request.language)
    # The TextPreprocessor.preprocess now includes the (placeholder) coreference resolution call if uncommented by user.
    lemmatized = TextPreprocessor.preprocess(request.text, lang_str)
    return PreprocessResponse(raw_text=request.text, language=request.language, lemmatized_text=lemmatized)


@app.post("/vectorize/", response_model=VectorizeResponse, tags=["Processing"])
async def vectorize_text_endpoint(request: TextProcessRequest):
    lang_str = get_lang_str(request.language)
    lemmatized = TextPreprocessor.preprocess(request.text, lang_str)
    if not lemmatized.strip(): raise HTTPException(status_code=400, detail="Empty after preprocessing.")
    vectorizer = get_vectorizer(request.language)
    vector_np = vectorizer.transform([lemmatized])[0]
    return VectorizeResponse(raw_text=request.text, language=request.language, lemmatized_text=lemmatized,
                             vector=vector_np.tolist())


# --- Words API (Updated for Triplet Extraction) ---
@app.post("/words/", response_model=WordResponse, status_code=201, tags=["Words"])
async def create_word_endpoint(word_data: WordCreateRequest, session: Session = Depends(get_session)):
    lang_str = get_lang_str(word_data.language)
    word_raw = word_data.raw_text.strip()
    if not word_raw: raise HTTPException(status_code=400, detail="Word raw_text empty.")

    # Lemmatized word (for ID and general processing) using TextPreprocessor
    word_lemmatized_for_id = TextPreprocessor.preprocess(word_raw, lang_str)
    if not word_lemmatized_for_id: raise HTTPException(status_code=400, detail="Word empty after TextPreprocessor.")
    word_id = generate_sha256_id(f"{word_lemmatized_for_id}_{lang_str}")

    if not session.get(Topic, word_data.topic_id): raise HTTPException(status_code=404, detail="Topic not found.")
    if session.get(Word, word_id): raise HTTPException(status_code=409, detail="Word already exists.")

    db_word = Word(id=word_id, topic_id=word_data.topic_id, raw_text=word_raw,
                   lemmatized_text=word_lemmatized_for_id,  # This is from TextPreprocessor
                   first_letter=word_lemmatized_for_id[0].lower() if word_lemmatized_for_id else None,
                   language=lang_str, info=word_data.info)
    session.add(db_word)

    desc_raw = word_data.description_raw_text.strip()  # This text goes to TripleExtractor
    if not desc_raw: raise HTTPException(status_code=400, detail="Description raw_text empty.")

    # Lemmatized description (for vectorization and ID) using TextPreprocessor
    desc_lemmatized_for_vec = TextPreprocessor.preprocess(desc_raw, lang_str)
    if not desc_lemmatized_for_vec: raise HTTPException(status_code=400,
                                                        detail="Description empty after TextPreprocessor.")
    desc_id = generate_sha256_id(f"{desc_lemmatized_for_vec}_{lang_str}")

    db_description = session.get(Description, desc_id)
    if db_description and db_description.word_id != word_id:
        raise HTTPException(status_code=409, detail="Desc content exists for another word.")
    if not db_description:
        db_description = Description(id=desc_id, word_id=word_id, raw_text=desc_raw,
                                     lemmatized_text=desc_lemmatized_for_vec,  # From TextPreprocessor
                                     language=lang_str)
        session.add(db_description)

    if not session.exec(select(Embedding).where(Embedding.description_id == desc_id)).first():
        vectorizer = get_vectorizer(word_data.language)
        desc_vector_np = vectorizer.transform([desc_lemmatized_for_vec])[0]  # Vectorize TextPreprocessor output
        db_embedding = Embedding(description_id=desc_id, embedding=desc_vector_np.tolist(), language=lang_str)
        session.add(db_embedding)

    # Triplet Generation using actual_extract_spo_from_description
    # Pass raw description text (desc_raw) to the new extractor.
    extracted_triplets_data = actual_extract_spo_from_description(desc_raw, lang_str, word_id, desc_id)

    for triplet_data in extracted_triplets_data:
        # triplet_data keys: subject_id, subject_type, predicate_raw, predicate_lemma, object_literal_raw, object_literal_lemma, language, source_sentence, info
        pred_raw = triplet_data["predicate_raw"]
        pred_lemma = triplet_data.get("predicate_lemma")
        obj_lit_raw = triplet_data.get("object_literal_raw")
        obj_lit_lemma = triplet_data.get("object_literal_lemma")

        # ID for triplet: sha256(subject_id + predicate_lemma_or_raw + object_lemma_or_raw + lang)
        pred_part_hash = pred_lemma if pred_lemma else pred_raw
        obj_part_hash = obj_lit_lemma if obj_lit_lemma else obj_lit_raw

        triplet_content_for_hash = f"{word_id}_{pred_part_hash}_{obj_part_hash}_{lang_str}"
        triplet_db_id = generate_sha256_id(triplet_content_for_hash)

        if not session.get(Triplet, triplet_db_id):
            db_triplet = Triplet(
                id=triplet_db_id,
                subject_id=word_id,  # From context
                subject_type="word",
                predicate_raw=pred_raw,
                predicate_lemma=pred_lemma,
                object_literal_raw=obj_lit_raw,
                object_literal_lemma=obj_lit_lemma,
                language=lang_str,
                source_sentence=triplet_data.get("source_sentence"),
                info=triplet_data.get("info")
            )
            session.add(db_triplet)

    session.commit()
    session.refresh(db_word)
    if db_description and db_description not in session: session.refresh(db_description)

    final_descriptions = session.exec(select(Description).where(Description.word_id == word_id)).all()
    final_triplets = session.exec(select(Triplet).where(Triplet.subject_id == word_id)).all()
    return WordResponse.model_validate(db_word, update={'descriptions': final_descriptions, 'triplets': final_triplets})


@app.get("/words/", response_model=List[WordResponse], tags=["Words"])
async def get_words_endpoint(first_letter: Optional[str] = Query(None, max_length=1),
                             topic_id: Optional[str] = Query(None), lang: Optional[Lang] = Query(None),
                             q: Optional[str] = Query(None), page: int = Query(1, ge=1),
                             size: int = Query(10, ge=1, le=100), session: Session = Depends(get_session)):
    stmt = select(Word)
    if first_letter: stmt = stmt.where(Word.first_letter == first_letter.lower())
    if topic_id: stmt = stmt.where(Word.topic_id == topic_id)
    if lang: stmt = stmt.where(Word.language == get_lang_str(lang))
    if q: stmt = stmt.where(or_(Word.raw_text.ilike(f"%{q}%"), Word.lemmatized_text.ilike(f"%{q}%")))
    words_db = session.exec(stmt.offset((page - 1) * size).limit(size).order_by(Word.raw_text)).all()
    return [WordResponse.model_validate(w, update={
        'descriptions': session.exec(select(Description).where(Description.word_id == w.id)).all(),
        'triplets': session.exec(select(Triplet).where(Triplet.subject_id == w.id)).all()
    }) for w in words_db]


@app.get("/words/{word_id}", response_model=WordResponse, tags=["Words"])
async def get_word_endpoint(word_id: str, session: Session = Depends(get_session)):
    db_word = session.get(Word, word_id)
    if not db_word: raise HTTPException(status_code=404, detail="Word not found")
    descs = session.exec(select(Description).where(Description.word_id == db_word.id)).all()
    trips = session.exec(select(Triplet).where(Triplet.subject_id == db_word.id)).all()
    return WordResponse.model_validate(db_word, update={'descriptions': descs, 'triplets': trips})


def normalize_vector(vector: List[float], target_dim: int = DEFAULT_VECTOR_DIM) -> List[float]:
    if len(vector) == target_dim:
        return vector
    elif len(vector) < target_dim:
        return vector + [0.0] * (target_dim - len(vector))
    else:
        return vector[:target_dim]


@app.put("/words/{word_id}", response_model=WordResponse, tags=["Words"])
async def update_word_endpoint(word_id: str, word_data: WordCreateRequest, session: Session = Depends(get_session)):
    # Fetch existing word
    db_word = session.get(Word, word_id)
    if not db_word:
        raise HTTPException(status_code=404, detail="Word not found")

    # Validate language
    lang_str = get_lang_str(word_data.language)
    if db_word.language != lang_str:
        raise HTTPException(status_code=400, detail="Cannot change language.")

    # Update word fields
    db_word.topic_id = word_data.topic_id
    db_word.info = word_data.info
    new_raw_word_text = word_data.raw_text.strip()
    if db_word.raw_text != new_raw_word_text:
        new_lem_word = TextPreprocessor.preprocess(new_raw_word_text, lang_str)
        if not new_lem_word:
            raise HTTPException(status_code=400, detail="New word raw_text empty after preprocess.")
        new_word_id = generate_sha256_id(f"{new_lem_word}_{lang_str}")
        if new_word_id != word_id:
            raise HTTPException(status_code=400, detail="Raw_text change alters ID. Not allowed.")
        db_word.raw_text = new_raw_word_text
        db_word.lemmatized_text = new_lem_word
        db_word.first_letter = new_lem_word[0].lower() if new_lem_word else None

    # Description update logic
    new_desc_raw_text = word_data.description_raw_text.strip()
    new_desc_lem_for_vec = TextPreprocessor.preprocess(new_desc_raw_text, lang_str)
    if not new_desc_lem_for_vec:
        raise HTTPException(status_code=400, detail="New description empty after preprocess.")

    # Generate new description ID
    new_desc_id = generate_sha256_id(f"{new_desc_lem_for_vec}_{lang_str}")
    if session.get(Description, new_desc_id) and session.get(Description, new_desc_id).word_id != word_id:
        raise HTTPException(status_code=409, detail="New description content exists for another word.")

    # Check existing descriptions
    current_descriptions = session.exec(select(Description).where(Description.word_id == word_id)).all()

    # Delete existing description, embedding, and triplets if description changed
    if current_descriptions:
        old_description = current_descriptions[0]  # Assuming one description
        if old_description.raw_text != new_desc_raw_text:
            logger.info(f"Updating description for word {word_id}")
            session.exec(delete(Embedding).where(Embedding.description_id == old_description.id))
            session.delete(old_description)
            session.exec(delete(Triplet).where(Triplet.subject_id == word_id))
    else:
        logger.info(f"No existing description for word {word_id}, creating new one.")

    # Create new description
    db_new_description = Description(
        id=new_desc_id,
        word_id=word_id,
        raw_text=new_desc_raw_text,
        lemmatized_text=new_desc_lem_for_vec,
        language=lang_str,
        created_at=datetime.now(timezone.utc)
    )
    session.add(db_new_description)

    # Generate and normalize embedding
    vec = get_vectorizer(word_data.language).transform([new_desc_lem_for_vec])[0]
    normalized_vec = normalize_vector(vec.tolist(), DEFAULT_VECTOR_DIM)
    session.add(Embedding(
        description_id=new_desc_id,
        embedding=normalized_vec,
        language=lang_str,
        created_at=datetime.now(timezone.utc)
    ))

    # Regenerate triplets
    new_triplets_data = actual_extract_spo_from_description(new_desc_raw_text, lang_str, word_id, new_desc_id)
    for t_data in new_triplets_data:
        pred_r, pred_l = t_data["predicate_raw"], t_data.get("predicate_lemma")
        obj_r, obj_l = t_data.get("object_literal_raw"), t_data.get("object_literal_lemma")
        pred_h = pred_l if pred_l else pred_r
        obj_h = obj_l if obj_l else obj_r
        t_db_id = generate_sha256_id(f"{word_id}_{pred_h}_{obj_h}_{lang_str}")
        with session.no_autoflush:  # Prevent premature flush
            if not session.get(Triplet, t_db_id):
                session.add(Triplet(
                    id=t_db_id,
                    subject_id=word_id,
                    predicate_raw=pred_r,
                    predicate_lemma=pred_l,
                    object_literal_raw=obj_r,
                    object_literal_lemma=obj_l,
                    language=lang_str,
                    source_sentence=t_data.get("source_sentence"),
                    info=t_data.get("info"),
                    created_at=datetime.now(timezone.utc)
                ))

    # Commit and refresh
    session.add(db_word)
    session.commit()
    session.refresh(db_word)
    final_descs = session.exec(select(Description).where(Description.word_id == db_word.id)).all()
    final_triplets = session.exec(select(Triplet).where(Triplet.subject_id == db_word.id)).all()
    return WordResponse.model_validate(db_word, update={'descriptions': final_descs, 'triplets': final_triplets})


def _delete_word_data(word_id: str, session: Session, commit_changes: bool = True):
    """Internal helper to delete all data associated with a word."""
    db_word = session.get(Word, word_id)
    if not db_word: return  # Already deleted or never existed

    descriptions = session.exec(select(Description).where(Description.word_id == word_id)).all()
    for desc in descriptions:
        session.exec(delete(Embedding).where(Embedding.description_id == desc.id))
        session.delete(desc)

    # Delete triplets where word is subject OR object (PDF section 5)
    session.exec(delete(Triplet).where(or_(
        (Triplet.subject_id == word_id) & (Triplet.subject_type == "word"),  # Explicitly subject_type word
        Triplet.object_id == word_id
    )))

    session.delete(db_word)
    if commit_changes:
        session.commit()
    logger.info(f"Data for word_id {word_id} deleted.")


@app.delete("/words/{word_id}", tags=["Words"])
async def delete_word_endpoint(word_id: str, session: Session = Depends(get_session)):
    db_word = session.get(Word, word_id)
    if not db_word: raise HTTPException(status_code=404, detail="Word not found")
    _delete_word_data(word_id, session)  # Use helper
    return {"ok": True, "message": f"Word '{db_word.raw_text}' and associated data deleted."}


# --- Descriptions API (Mainly GET, update is via PUT /words) ---
@app.get("/descriptions/{description_id}", response_model=DescriptionResponse, tags=["Descriptions"])
async def get_description_endpoint(description_id: str, session: Session = Depends(get_session)):
    desc = session.get(Description, description_id)
    if not desc: raise HTTPException(status_code=404, detail="Description not found")
    return desc


# --- Triplets API (Manual Creation) ---
@app.post("/triplets/", response_model=TripletResponse, status_code=201, tags=["Triplets (S-P-O)"])
async def create_manual_triplet_endpoint(triplet_data: TripletCreateRequest, session: Session = Depends(get_session)):
    lang_str = get_lang_str(triplet_data.language)

    # Subject must be an existing word
    subject_word = session.get(Word, triplet_data.subject_id)
    if not subject_word: raise HTTPException(status_code=404,
                                             detail=f"Subject word '{triplet_data.subject_id}' not found.")
    if subject_word.language != lang_str: raise HTTPException(status_code=400, detail="Subject word language mismatch.")

    if triplet_data.object_id and triplet_data.object_literal_raw:
        raise HTTPException(status_code=400, detail="Provide object_id OR object_literal_raw, not both.")

    obj_lit_lemma_val = triplet_data.object_literal_lemma
    if triplet_data.object_literal_raw and not obj_lit_lemma_val:  # If raw is given but lemma isn't, try to lemmatize
        if lang_str == Lang.Russian.value and russian_triple_extractor:  # Use TripleExtractor's lemmatizer
            obj_lit_lemma_val = russian_triple_extractor._lemmatize_token_text(triplet_data.object_literal_raw)
        else:  # Fallback for other languages or if Natasha not available
            obj_lit_lemma_val = TextPreprocessor.preprocess(triplet_data.object_literal_raw, lang_str)

    pred_lemma_val = triplet_data.predicate_lemma
    if not pred_lemma_val and triplet_data.predicate_raw:  # Auto-fill predicate_lemma if not provided
        if lang_str == Lang.Russian.value and russian_triple_extractor:
            pred_lemma_val = russian_triple_extractor._lemmatize_token_text(triplet_data.predicate_raw)
        else:
            pred_lemma_val = TextPreprocessor.preprocess(triplet_data.predicate_raw, lang_str)

    if triplet_data.object_id:
        obj_word = session.get(Word, triplet_data.object_id)
        if not obj_word: raise HTTPException(status_code=404,
                                             detail=f"Object word '{triplet_data.object_id}' not found.")
        if obj_word.language != lang_str: raise HTTPException(status_code=400, detail="Object word language mismatch.")
        object_part_hash = triplet_data.object_id
    else:
        object_part_hash = obj_lit_lemma_val if obj_lit_lemma_val else triplet_data.object_literal_raw

    pred_part_hash = pred_lemma_val if pred_lemma_val else triplet_data.predicate_raw

    triplet_db_id = generate_sha256_id(f"{triplet_data.subject_id}_{pred_part_hash}_{object_part_hash}_{lang_str}")
    if session.get(Triplet, triplet_db_id): raise HTTPException(status_code=409, detail="Triplet already exists.")

    db_triplet = Triplet(id=triplet_db_id, subject_id=triplet_data.subject_id, subject_type="word",
                         predicate_raw=triplet_data.predicate_raw, predicate_lemma=pred_lemma_val,
                         object_id=triplet_data.object_id, object_type="word" if triplet_data.object_id else None,
                         object_literal_raw=triplet_data.object_literal_raw, object_literal_lemma=obj_lit_lemma_val,
                         language=lang_str, source_sentence=triplet_data.source_sentence,
                         info=triplet_data.info or "Manually created")
    session.add(db_triplet);
    session.commit();
    session.refresh(db_triplet)
    return db_triplet


@app.get("/triplets/", response_model=List[TripletResponse], tags=["Triplets (S-P-O)"])
async def get_triplets_endpoint(subject_id: Optional[str] = Query(None), predicate: Optional[str] = Query(None),
                                object_id: Optional[str] = Query(None), object_literal_q: Optional[str] = Query(None),
                                lang: Optional[Lang] = Query(None), session: Session = Depends(get_session)):
    stmt = select(Triplet)
    if subject_id: stmt = stmt.where(Triplet.subject_id == subject_id)
    if predicate: stmt = stmt.where(
        or_(Triplet.predicate_raw.ilike(f"%{predicate}%"), Triplet.predicate_lemma.ilike(f"%{predicate}%")))
    if object_id: stmt = stmt.where(Triplet.object_id == object_id)
    if object_literal_q: stmt = stmt.where(or_(Triplet.object_literal_raw.ilike(f"%{object_literal_q}%"),
                                               Triplet.object_literal_lemma.ilike(f"%{object_literal_q}%")))
    if lang: stmt = stmt.where(Triplet.language == get_lang_str(lang))
    return session.exec(stmt.order_by(Triplet.created_at.desc())).all()


@app.delete("/triplets/{triplet_id}", tags=["Triplets (S-P-O)"])
async def delete_triplet_endpoint(triplet_id: str, session: Session = Depends(get_session)):
    db_triplet = session.get(Triplet, triplet_id)
    if not db_triplet: raise HTTPException(status_code=404, detail="Triplet not found")
    session.delete(db_triplet);
    session.commit()
    return {"ok": True, "message": "Triplet deleted"}


# --- Search API (Largely unchanged, ensure WordResponse includes triplets) ---
@app.get("/search/words", response_model=List[WordResponse], tags=["Search"])
async def search_words_endpoint(q: str, lang: Optional[Lang] = Query(None), topic_id: Optional[str] = Query(None),
                                page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=100),
                                session: Session = Depends(get_session)):
    stmt = select(Word);
    conditions = []
    q_processed = q;
    lang_str_proc = get_lang_str(lang) if lang else None
    if lang_str_proc:
        try:
            proc_cand = TextPreprocessor.preprocess(q, lang_str_proc);
            q_processed = proc_cand if proc_cand else q
        except Exception:
            pass  # Keep raw q

    txt_cond = or_(Word.raw_text.ilike(f"%{q}%"), Word.lemmatized_text.ilike(f"%{q_processed}%"))
    if q_processed != q: txt_cond = or_(txt_cond, Word.lemmatized_text.ilike(f"%{q}%"))
    conditions.append(txt_cond)

    if lang: conditions.append(Word.language == get_lang_str(lang))
    if topic_id: conditions.append(Word.topic_id == topic_id)
    stmt = stmt.where(*conditions)

    words_db = session.exec(stmt.offset((page - 1) * size).limit(size).order_by(Word.raw_text)).all()
    return [WordResponse.model_validate(w, update={
        'descriptions': session.exec(select(Description).where(Description.word_id == w.id)).all(),
        'triplets': session.exec(select(Triplet).where(Triplet.subject_id == w.id)).all()
    }) for w in words_db]


@app.get("/search/similar", response_model=List[WordResponse], tags=["Search"])
async def search_similar_words_by_description_vector(description_id: str, top_k: int = Query(5, ge=1, le=50),
                                                     session: Session = Depends(get_session)):
    source_emb_obj = session.exec(select(Embedding).where(Embedding.description_id == description_id)).first()
    if not source_emb_obj: raise HTTPException(status_code=404, detail="Source description embedding not found.")
    source_vector = np.array(source_emb_obj.embedding);
    source_lang = source_emb_obj.language

    # Using <-> (L2 distance) operator from pgvector
    # This query gets Embedding entities and their distance to the source_vector
    similar_embeddings_stmt = (
        select(Embedding, Embedding.embedding.l2_distance(source_vector).label("distance"))
        .where(Embedding.language == source_lang)
        .where(Embedding.description_id != description_id)
        .order_by("distance")  # pgvector will use index if available
        .limit(top_k)
    )
    results = session.exec(similar_embeddings_stmt).all()

    similar_words_responses = []
    for emb_obj, dist_val in results:
        desc = session.get(Description, emb_obj.description_id)
        if not desc: continue
        word_obj = session.get(Word, desc.word_id)
        if not word_obj: continue

        word_descs = session.exec(select(Description).where(Description.word_id == word_obj.id)).all()
        word_triplets = session.exec(select(Triplet).where(Triplet.subject_id == word_obj.id)).all()
        # Add distance to response if WordResponse model is extended
        word_resp = WordResponse.model_validate(word_obj,
                                                update={'descriptions': word_descs, 'triplets': word_triplets})
        # If you want to add distance, you might need to modify WordResponse or return a different model
        # For example: response_dict = word_resp.model_dump(); response_dict['distance'] = float(dist_val)
        similar_words_responses.append(word_resp)

    return similar_words_responses


def convert_numpy_types(obj):
    """
    Recursively converts NumPy types to Python native types for JSON serialization.

    Args:
        obj: Object to convert (dict, list, tuple, or scalar).

    Returns:
        Object with NumPy types converted to Python types.
    """
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


def load_graph_from_db(language: str, session: Session) -> nx.DiGraph:
    """Loads the knowledge graph from graph_nodes and graph_edges tables."""
    graph = nx.DiGraph()

    # Load nodes
    nodes = session.exec(
        select(GraphNode).where(GraphNode.language == language)
    ).all()
    for node in nodes:
        graph.add_node(
            node.term,
            word_id=node.word_id,
            centroid_weight=node.centroid_weight,
            centrality_weight=node.centrality_weight,
            combined_weight=node.combined_weight
        )

    # Load edges
    edges = session.exec(
        select(GraphEdge).where(GraphEdge.language == language)
    ).all()
    for edge in edges:
        source_term = session.exec(
            select(GraphNode.term).where(GraphNode.id == edge.source_node_id)
        ).first()
        target_term = session.exec(
            select(GraphNode.term).where(GraphNode.id == edge.target_node_id)
        ).first()
        if source_term and target_term:
            graph.add_edge(
                source_term,
                target_term,
                type=edge.type,
                predicate=edge.predicate,
                weight=edge.weight,
                triplet_id=edge.triplet_id
            )

    logger.info(f"Loaded graph from DB: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


@app.get("/graph/", tags=["Knowledge Graph"])
async def get_knowledge_graph(language: Lang = Lang.Russian, rebuild: bool = False,
                              session: Session = Depends(get_session)):
    if rebuild:
        graph = build_knowledge_graph(language)
    else:
        graph = load_graph_from_db(language.value, session)
        if graph.number_of_nodes() == 0:
            logger.info("No graph found in DB, building new graph")
            graph = build_knowledge_graph(language)

    nodes_data = [{"node": node, **convert_numpy_types(data)} for node, data in graph.nodes(data=True)]
    edges_data = [{"source": u, "target": v, **convert_numpy_types(data)} for u, v, data in graph.edges(data=True)]
    return {
        "nodes": nodes_data,
        "edges": edges_data,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges()
    }


@app.get("/graph/visualize/", tags=["Knowledge Graph"])
async def visualize_knowledge_graph_endpoint(language: Lang = Lang.Russian):
    output_file = f"knowledge_graph_{language.value}.png"
    visualize_knowledge_graph(language, output_file)
    return {"message": f"Graph visualization saved to {output_file}"}


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("stemming.api:app", host="0.0.0.0", port=8000, reload=True)
