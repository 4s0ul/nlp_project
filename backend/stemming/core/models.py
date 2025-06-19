import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass
from fastclasses_json import dataclass_json
from typing import List, Optional, Generator
import numpy as np

from fastapi import FastAPI, Depends, HTTPException, Query
from natasha.doc import DocToken
from sqlmodel import Field, SQLModel, create_engine, Session, select, delete, Column, or_, Relationship
from sqlalchemy import text as sa_text, TEXT  # For executing raw SQL if needed
from pgvector.sqlalchemy import Vector

from stemming.core.vectorization import DEFAULT_VECTOR_DIM


class Lang(Enum):
    Russian = "russian"
    English = "english"

def generate_sha256_id(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class Topic(SQLModel, table=True):
    __tablename__ = "topics"
    id: str = Field(primary_key=True, max_length=64, index=True)
    name: str = Field(sa_type=TEXT, unique=True)
    info: Optional[str] = Field(default=None, sa_type=TEXT)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Word(SQLModel, table=True):
    __tablename__ = "words"
    id: str = Field(primary_key=True, max_length=64, index=True)
    topic_id: str = Field(foreign_key="topics.id", max_length=64, index=True)
    raw_text: str = Field(sa_type=TEXT)
    cleaned_text: Optional[str] = Field(default=None, sa_type=TEXT)
    lemmatized_text: str = Field(sa_type=TEXT, index=True)
    first_letter: Optional[str] = Field(default=None, max_length=1, index=True)
    language: str = Field(max_length=10, index=True)
    info: Optional[str] = Field(default=None, sa_type=TEXT)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    graph_nodes: List["GraphNode"] = Relationship(back_populates="word")


class Description(SQLModel, table=True):
    __tablename__ = "descriptions"
    id: str = Field(primary_key=True, max_length=64, index=True)
    word_id: str = Field(foreign_key="words.id", max_length=64, index=True)
    raw_text: str = Field(sa_type=TEXT)
    cleaned_text: Optional[str] = Field(default=None, sa_type=TEXT)
    lemmatized_text: str = Field(sa_type=TEXT, index=True)
    language: str = Field(max_length=10, index=True)
    info: Optional[str] = Field(default=None, sa_type=TEXT)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Embedding(SQLModel, table=True):
    __tablename__ = "embeddings"
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    description_id: str = Field(foreign_key="descriptions.id", max_length=64, unique=True, index=True)
    embedding: List[float] = Field(sa_column=Column(Vector(DEFAULT_VECTOR_DIM)))  # Assuming DEFAULT_VECTOR_DIM = 1536
    language: str = Field(max_length=10, index=True)
    info: Optional[str] = Field(default=None, sa_type=TEXT)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Triplet(SQLModel, table=True):
    __tablename__ = "triplets"
    id: str = Field(primary_key=True, max_length=64, index=True)
    subject_id: str = Field(max_length=64, index=True)
    subject_type: str = Field(max_length=20, index=True, default="word")
    predicate_raw: str = Field(sa_type=TEXT, index=True)
    predicate_lemma: Optional[str] = Field(default=None, sa_type=TEXT, index=True)
    object_id: Optional[str] = Field(default=None, max_length=64, index=True)
    object_type: Optional[str] = Field(default=None, max_length=20)
    object_literal_raw: Optional[str] = Field(default=None, sa_type=TEXT)
    object_literal_lemma: Optional[str] = Field(default=None, sa_type=TEXT, index=True)
    source_sentence: Optional[str] = Field(default=None, sa_type=TEXT)
    language: str = Field(max_length=10, index=True)
    info: Optional[str] = Field(default=None, sa_type=TEXT)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    graph_edges: List["GraphEdge"] = Relationship(back_populates="triplet")


class GraphNode(SQLModel, table=True):
    __tablename__ = "graphnode"
    id: str = Field(default_factory=lambda: generate_sha256_id(str(uuid.uuid4())), primary_key=True)
    word_id: str = Field(foreign_key="words.id")
    term: str
    language: str
    centroid_weight: float
    centrality_weight: float
    combined_weight: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    word: Word = Relationship()


class GraphEdge(SQLModel, table=True):
    __tablename__ = "graphedge"
    id: str = Field(default_factory=lambda: generate_sha256_id(str(uuid.uuid4())), primary_key=True)
    source_node_id: str = Field(foreign_key="graphnode.id")
    target_node_id: str = Field(foreign_key="graphnode.id")
    type: str  # e.g., 'triplet', 'semantic', 'hierarchical'
    predicate: Optional[str] = None  # For triplet edges
    weight: float
    language: str
    triplet_id: Optional[str] = Field(default=None, foreign_key="triplets.id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_node: GraphNode = Relationship(sa_relationship_kwargs={"foreign_keys": "GraphEdge.source_node_id"})
    target_node: GraphNode = Relationship(sa_relationship_kwargs={"foreign_keys": "GraphEdge.target_node_id"})
    triplet: Optional[Triplet] = Relationship()


class TopicResponse(SQLModel):
    id: Optional[str] = None
    name: str
    info: Optional[str]
    created_at: datetime


class WordCreateRequest(SQLModel):
    topic_id: str
    raw_text: str
    cleaned_text: Optional[str] = None
    lemmatized_text: Optional[str] = None
    description_raw_text: str  # This will be used for triplet extraction
    description_cleaned_text: Optional[str] = None
    description_lemmatized_text: Optional[str] = None
    language: Lang
    info: Optional[str] = None


class EmbeddingResponse(SQLModel):
    id: Optional[int] = None
    description_id: str
    embedding: List[float]
    language: str
    info: Optional[str] = None
    created_at: datetime


class DescriptionResponse(SQLModel):
    id: str
    word_id: str
    raw_text: str
    cleaned_text: Optional[str] = None
    lemmatized_text: str
    language: str
    info: Optional[str] = None
    created_at: datetime
    embeddings: List[EmbeddingResponse] = []


class TripletResponse(SQLModel):
    id: str
    subject_id: str
    subject_type: str
    predicate_raw: str
    predicate_lemma: Optional[str] = None
    object_id: Optional[str] = None
    object_type: Optional[str] = None
    object_literal_raw: Optional[str] = None
    object_literal_lemma: Optional[str] = None
    source_sentence: Optional[str] = None
    language: str
    info: Optional[str] = None
    created_at: datetime


class WordResponse(SQLModel):
    id: str
    topic_id: str
    raw_text: str
    cleaned_text: Optional[str] = None
    lemmatized_text: str
    first_letter: Optional[str] = None
    language: str
    info: Optional[str] = None
    created_at: datetime
    descriptions: List[DescriptionResponse] = []
    triplets: List[TripletResponse] = []


class TripletCreateRequest(SQLModel):  # For manual triplet creation
    subject_id: str  # Must be an existing Word ID for this manual endpoint
    # subject_type: str = Query("word", pattern="^(word)$") # For manual, let's assume subject is always a Word for simplicity, matching auto-extraction.
    predicate_raw: str
    predicate_lemma: Optional[str] = None
    object_literal_raw: Optional[str] = None
    object_literal_lemma: Optional[str] = None  # Will be auto-filled if object_literal_raw is given and lemma is None
    object_id: Optional[str] = None  # Link to another Word ID as object
    language: Lang
    source_sentence: Optional[str] = None
    info: Optional[str] = None


class TextProcessRequest(SQLModel): text: str; language: Lang


class PreprocessResponse(SQLModel): raw_text: str; language: Lang; lemmatized_text: str


class VectorizeResponse(SQLModel): raw_text: str; language: Lang; lemmatized_text: str; vector: List[float]


DATABASE_URL = "postgresql+psycopg2://user:password@stemming-db:5432/stemming_db"
engine = create_engine(DATABASE_URL)


def create_db_and_tables():
    with engine.connect() as connection:
        connection.execute(sa_text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.commit()
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session: yield session
