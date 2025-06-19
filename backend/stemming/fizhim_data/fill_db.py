import hashlib
from datetime import datetime
import psycopg2
from psycopg2 import sql
import pandas as pd


# Подключение к базе данных
def get_db_connection():
    conn = psycopg2.connect(
        host="localhost", database="stemming_db", user="user", password="password"
    )
    return conn


# Функция для создания SHA-256 хэша
def generate_hash(text):
    return hashlib.sha256(text.encode("utf-8-sig")).hexdigest()


# Основная функция для импорта данных
def import_data_from_csv(
    csv_original, csv_stopwords, csv_lemma, csv_vectors, csv_triplets
):
    conn = get_db_connection()
    cur = conn.cursor()

    topic_id = generate_hash("Физико-химические Термины")

    cur.execute(
        sql.SQL("""
        INSERT INTO topics (id, name, info, created_at) 
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
        """),
        (topic_id, "Физико-химические Термины", "Topic for Физико-химические Термины terms", datetime.now()),
    )

    df_original = pd.read_csv(csv_original, encoding="utf-8", sep=";")
    df_stopwrods = pd.read_csv(csv_stopwords, encoding="utf-8", sep=";")
    df_lemma = pd.read_csv(csv_lemma, encoding="utf-8", sep=";")
    df_vectors = pd.read_csv(csv_vectors, encoding="utf-8", sep=";")
    df_triplets = pd.read_csv(csv_triplets, encoding="utf-8", sep=";")

    df_total = pd.concat([df_original, df_stopwrods, df_lemma, df_vectors], axis=1)

    for index, row in df_total.iterrows():
        en_term_orig = row["en_term"]
        en_def_orig = row["en_definition"]
        ru_term_orig = row["ru_term"]
        ru_def_orig = row["ru_definition"]

        en_term_stop = row["en_term_no_stopwords"]
        en_def_stop = row["en_definition_no_stopwords"]
        ru_term_stop = row["ru_term_no_stopwords"]
        ru_def_stop = row["ru_definition_no_stopwords"]

        en_term_proc = row["en_term_processed"]
        en_def_proc = row["en_definition_processed"]
        ru_term_proc = row["ru_term_processed"]
        ru_def_proc = row["ru_definition_processed"]

        en_def_vect = row["en_definition_vector"]
        ru_def_vect = row["ru_definition_vector"]

        process_term(
            cur,
            topic_id,
            en_term_orig,
            en_def_orig,
            en_term_stop,
            en_def_stop,
            en_term_proc,
            en_def_proc,
            en_def_vect,
        )
        process_term(
            cur,
            topic_id,
            ru_term_orig,
            ru_def_orig,
            ru_term_stop,
            ru_def_stop,
            ru_term_proc,
            ru_def_proc,
            ru_def_vect,
        )

    conn.commit()

    for index, row in df_triplets.iterrows():
        obj = row["object"]
        pred = row["relation"]
        subj = row["subject"]

        process_triplets(cur, obj, subj, pred)

    conn.commit()
    cur.close()
    conn.close()
    print("Импорт данных завершен успешно!")


def process_term(
    cur, topic_id, term, definition, term_stop, def_stop, term_proc, def_proc, def_vect
):
    # Вставляем слово
    print(term)
    word_id = generate_hash(f"{term}")

    cur.execute(
        sql.SQL("""
        INSERT INTO words (id, topic_id, raw_text, cleaned_text, lemmatized_text, first_letter, language, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, 'russian', %s)
        ON CONFLICT (id) DO NOTHING
        """),
        (word_id, topic_id, term, term_stop, term_proc, term[:1], datetime.now()),
    )

    # Если есть определение, вставляем описание
    description_id = generate_hash(f"{definition}")

    cur.execute(
        sql.SQL("""
        INSERT INTO descriptions (id, word_id, raw_text, cleaned_text, lemmatized_text, language, created_at)
        VALUES (%s, %s, %s, %s, %s, 'russian', %s)
        ON CONFLICT (id) DO NOTHING
        """),
        (description_id, word_id, definition, def_stop, def_proc, datetime.now()),
    )

    cur.execute(
        sql.SQL("""
        INSERT INTO embeddings (description_id, embedding, language, created_at)
        VALUES (%s, %s, 'russian', %s)
        """),
        (description_id, def_vect, datetime.now()),
    )


def process_triplets(cur, object, subject, predicate):
    print(object)
    literal = 0
    cur.execute(
        sql.SQL("""
        SELECT id
        FROM words
        WHERE raw_text = %s
        """),
        (object,),
    )

    obj_id = cur.fetchone()
    if obj_id is None:
        cur.execute(
            sql.SQL("""
            SELECT id
            FROM descriptions
            WHERE raw_text = %s
            """),
            (object,),
        )
        obj_id = cur.fetchone()
        literal = 1

    cur.execute(
        sql.SQL("""
        SELECT id
        FROM words
        WHERE raw_text = %s
        """),
        (subject,),
    )

    subj_id = cur.fetchone()
    if subj_id is None:
        cur.execute(
            sql.SQL("""
            SELECT id
            FROM descriptions
            WHERE raw_text = %s
            """),
            (subject,),
        )
        subj_id = cur.fetchone()

    if obj_id and subj_id:
        id = generate_hash(f"{object + predicate + subject}")
        if literal == 1:
            cur.execute(
                sql.SQL("""
                INSERT INTO triplets (id, subject_id, subject_type, predicate_raw, object_literal, language, created_at)
                VALUES (%s, %s, 'word', %s, %s, 'russian', %s)
                """),
                (id, subj_id, predicate, obj_id, datetime.now()),
            )
        else:
            cur.execute(
                sql.SQL("""
                INSERT INTO triplets (id, subject_id, subject_type, predicate_raw, object_id, language, created_at)
                VALUES (%s, %s, 'word', %s, %s, 'russian', %s)
                """),
                (id, subj_id, predicate, obj_id, datetime.now()),
            )


# Запуск импорта
if __name__ == "__main__":
    csv_original = "Original.csv"  # Укажите путь к вашему CSV файлу
    csv_stopwords = "Stopwords.csv"
    csv_lemma = "Lemma.csv"
    csv_vectors = "Vectors.csv"
    csv_triplets = "Triplets.csv"
    import_data_from_csv(
        csv_original, csv_stopwords, csv_lemma, csv_vectors, csv_triplets
    )
