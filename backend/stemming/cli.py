
import argparse
from tqdm import tqdm
from stemming.core.io_utils import Terms, serialize_to_csv
from stemming.core.models import Lang
from stemming.core.vectorization import AdvancedWord2VecVectorizer, vectorize
from stemming.core.simple_graph_builder import GraphBuilder
from stemming.core.models import TermVector


def dictionary_from_terms(terms: Terms, language: Lang):
    return [TermVector.from_term(term, language) for term in terms.terms]

def process_term(terms: Terms, language: Lang, output_file: str):
    term_vectors = dictionary_from_terms(terms, language)
    texts = [tv.term + " " + tv.definition for tv in term_vectors]
    vectorizer = AdvancedWord2VecVectorizer(language="ru_core_news_sm" if language == Lang.Russian else "en_core_web_md")
    vectorizer.fit(texts)
    vectorized_terms = [vectorize(tv, language, vectorizer) for tv in tqdm(term_vectors, desc="Vectorizing")]
    graph = GraphBuilder(language)
    graph.build_graph(vectorized_terms)
    graph.visualize_graph()
    serialize_to_csv(vectorized_terms, output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Path to input JSON file")
    parser.add_argument("--with_english", action="store_true")
    args = parser.parse_args()

    terms = Terms(args.json_file)
    process_term(terms, Lang.Russian, "terms_vectorized_ru.csv")
    if args.with_english:
        process_term(terms, Lang.English, "terms_vectorized_en.csv")

if __name__ == "__main__":
    main()
