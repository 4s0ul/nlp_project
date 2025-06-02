import json
import csv
from pathlib import Path
from typing import List
from stemming.core.models import Term, TermVectorized
from loguru import logger

class Terms:
    def __init__(self, json_file: str):
        self.__terms: List[Term] = []
        self._load_terms(json_file)

    def _load_terms(self, json_file: str) -> None:
        file_path = Path(json_file)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")

        with file_path.open(encoding="utf-8") as f:
            terms_data = json.load(f)
            self.__terms = [Term.from_dict(term_dict) for term_dict in terms_data]

    @property
    def terms(self) -> List[Term]:
        return self.__terms.copy()

def serialize_to_csv(terms_vectorized: List[TermVectorized], output_file: str) -> None:
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Language", "Term", "Definition", "Term Metadata", "Term Vector", "Definition Vector"])

        for tv in terms_vectorized:
            writer.writerow([
                tv.language.value if tv.language else "",
                tv.term_metadata.term if tv.term_metadata else "",
                tv.term_metadata.definition if tv.term_metadata else "",
                tv.term_metadata.to_dict() if tv.term_metadata else {},
                tv.term.tolist() if tv.term is not None else [],
                tv.definition.tolist() if tv.definition is not None else []
            ])