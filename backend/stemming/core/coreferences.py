import re
from loguru import logger

import spacy
import nltk

from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)

# Initialize Natasha components (assuming models are downloaded)
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

# Define required morphology for common pronoun lemmas
# Based on hardcoded expectation, acknowledging morphology extraction issue in 1.6.0 output
PRONOUN_LEMMA_FEATURES = {
    'он': {'Gender': 'Masc', 'Number': 'Sing'},
    'она': {'Gender': 'Fem', 'Number': 'Sing'},
    'оно': {'Gender': 'Neut', 'Number': 'Sing'},
    'они': {'Number': 'Plur'}, # Gender doesn't apply to plural 'они'
    'его': {'RequiredGender': ['Masc', 'Neut'], 'Number': 'Sing'}, # 'его' can be masc or neut genitive/possessive
    'ее': {'Gender': 'Fem', 'Number': 'Sing'},
    'их': {'Number': 'Plur'},
}

# Define what qualifies as a potential antecedent and its key features
# In a real system, you'd use extracted morphology here if reliable
def get_antecedent_features(token, ner_tag, doc_tokens):
    """
    Extracts relevant features from a potential antecedent token/span.
    Limited extraction due to observed 1.6.0 morphology output.
    """
    features = {}
    # Check if it's a relevant POS or NER type
    if token.pos in ['NOUN', 'PROPN'] or ner_tag != 'O':
        features['POS'] = token.pos
        features['NER'] = ner_tag
        features['Lemma'] = token.lemma
        features['Text'] = token.text # Keep original text for replacement

        # --- Attempt to get Gender/Number from tag (if available in 1.6.0 output) ---
        # This part might still yield None based on previous output,
        # but we include the logic in case some tags are present.
        extracted_gender = None
        extracted_number = None
        if hasattr(token, 'tag') and token.tag:
             try:
                 tag_features = dict(item.split('=') for item in token.tag.split('|') if '=' in item)
                 extracted_gender = tag_features.get('Gender')
                 extracted_number = tag_features.get('Number')
             except Exception:
                  pass # Ignore parsing errors

        features['ExtractedGender'] = extracted_gender
        features['ExtractedNumber'] = extracted_number
        # --- End Morphology Extraction Attempt ---

    return features


def check_agreement(pronoun_lemma_features, antecedent_features):
    """
    Checks agreement between pronoun requirements and antecedent features.
    Simplified check due to potential missing extracted morphology.
    Relies more on POS, NER, and hardcoded pronoun expectations.
    """
    if not antecedent_features:
        return False

    p_gender_req = pronoun_lemma_features.get('Gender')
    p_gender_req_list = pronoun_lemma_features.get('RequiredGender') # For 'его'
    p_number_req = pronoun_lemma_features.get('Number')

    a_pos = antecedent_features.get('POS')
    a_ner = antecedent_features.get('NER')
    a_extracted_gender = antecedent_features.get('ExtractedGender')
    a_extracted_number = antecedent_features.get('ExtractedNumber')

    # Rule 1: Check Number agreement (Plural vs Singular) - this is often reliable
    if p_number_req == 'Plur':
        # Pronoun is plural ('они', 'их'). Antecedent must also be plural.
        # We rely on the *extracted* number here if available, or assume plural nouns are plural.
        if a_extracted_number == 'Plur' or (a_pos in ['NOUN', 'PROPN'] and antecedent_features.get('Lemma') and antecedent_features['Lemma'].endswith('ы')): # Very simple plural guess
             pass # Number agreement holds (if extracted or guessed)
        else:
             return False # Number mismatch
    elif p_number_req == 'Sing':
        # Pronoun is singular. Antecedent must be singular.
        if a_extracted_number == 'Sing' or (a_pos in ['NOUN', 'PROPN'] and not (antecedent_features.get('Lemma') and antecedent_features['Lemma'].endswith('ы'))): # Very simple singular guess
             pass # Number agreement holds (if extracted or guessed)
        else:
             return False # Number mismatch

    # Rule 2: Check Gender agreement for singular pronouns
    if p_number_req == 'Sing':
        if p_gender_req: # Specific gender required (Masc, Fem, Neut)
            # Check if extracted gender matches, OR if it's a PER entity and NER/POS align
            if a_extracted_gender == p_gender_req:
                pass # Extracted gender matches
            elif a_ner == 'PER':
                 # If it's a person NER, maybe the POS gives a hint or it's a proper noun
                 if p_gender_req == 'Masc' and (a_pos == 'PROPN' or antecedent_features.get('Lemma','').endswith(('ов', 'ев', 'ин'))): # Very rough guess for masc names
                      pass
                 elif p_gender_req == 'Fem' and (a_pos == 'PROPN' or antecedent_features.get('Lemma','').endswith(('а', 'я', 'ова', 'ева', 'ина'))): # Very rough guess for fem names
                      pass
                 else:
                      return False # Person NER doesn't seem to match required gender
            elif a_pos in ['NOUN', 'PROPN']:
                 # If a common/proper noun, very rough guess based on ending if morphology missing
                 if p_gender_req == 'Masc' and antecedent_features.get('Lemma','').endswith(('й', 'ь', 'р', 'л')): # Rough masc noun guess
                     pass
                 elif p_gender_req == 'Fem' and antecedent_features.get('Lemma','').endswith(('а', 'я')): # Rough fem noun guess
                     pass
                 elif p_gender_req == 'Neut' and antecedent_features.get('Lemma','').endswith(('о', 'е')): # Rough neut noun guess
                      pass
                 else:
                      return False # Noun doesn't seem to match required gender
            else:
                return False # No basis to check gender

        elif p_gender_req_list: # 'его' can be Masc or Neut
             if a_extracted_gender in p_gender_req_list:
                  pass # Extracted gender matches one of the requirements
             elif a_ner == 'PER':
                 # For 'его' referring to a person, assume Masc if it's a Person NER and not obviously female name
                 if 'Masc' in p_gender_req_list and (a_pos == 'PROPN' or not antecedent_features.get('Lemma','').endswith(('а', 'я'))):
                      pass
                 else:
                      return False # Person NER doesn't seem to match (e.g., 'его' referring to Мария)
             elif a_pos in ['NOUN', 'PROPN']:
                 # For 'его' referring to a noun, very rough guess
                 if ('Masc' in p_gender_req_list and antecedent_features.get('Lemma','').endswith(('й', 'ь', 'р', 'л'))) or \
                    ('Neut' in p_gender_req_list and antecedent_features.get('Lemma','').endswith(('о', 'е'))):
                     pass
                 else:
                     return False # Noun doesn't seem to match Masc/Neut requirement
             else:
                 return False # No basis to check gender

    # If we reached here, number agreement holds, and gender agreement holds for singular.
    return True


def resolve_coreference_with_natasha_rules(text):
    """
    A simple rule-based coreference resolver using Natasha v1.6.0 features.

    Args:
        text: Input string in Russian.

    Returns:
        A single string with simple coreferences potentially resolved by replacing
        pronouns with identified antecedents based on basic rules.
    """
    doc = Doc(text)

    # Apply pipeline components sequentially
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)

    # Lemmatization (needed for matching pronoun lemmas)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    resolved_tokens = []
    tokens_info = [] # Store info about each token for easier backward lookup

    # First pass: extract info for all tokens
    for i, token in enumerate(doc.tokens):
         ner_tag = 'O'
         for span in doc.spans:
              if token.start >= span.start and token.stop <= span.stop:
                   ner_tag = span.type
                   break

         gender = None
         number = None
         if hasattr(token, 'tag') and token.tag:
              try:
                  features = dict(item.split('=') for item in token.tag.split('|') if '=' in item)
                  gender = features.get('Gender')
                  number = features.get('Number')
              except Exception:
                   pass # Ignore parsing errors

         tokens_info.append({
             'index': i,
             'token': token,
             'lemma': token.lemma,
             'pos': token.pos,
             'ner': ner_tag,
             'extracted_gender': gender, # May be None in 1.6.0
             'extracted_number': number, # May be None in 1.6.0
             'text': token.text
         })


    # Second pass: resolve coreferences
    for i, token_info in enumerate(tokens_info):
        token = token_info['token']
        lemma = token_info['lemma']
        pos = token_info['pos']

        # Check if the token is a target pronoun based on its lemma and POS
        if pos == 'PRON' and lemma in PRONOUN_LEMMA_FEATURES:
            pronoun_reqs = PRONOUN_LEMMA_FEATURES[lemma]
            best_antecedent = None

            # Look backward for potential antecedents
            for j in range(i - 1, -1, -1):
                antecedent_info = tokens_info[j]
                antecedent_token = antecedent_info['token']
                antecedent_ner = antecedent_info['ner']
                antecedent_pos = antecedent_info['pos']

                # Define what qualifies as a potential antecedent token type
                is_potential_antecedent_type = (antecedent_pos in ['NOUN', 'PROPN'] or antecedent_ner in ['PER', 'ORG', 'LOC'])

                if is_potential_antecedent_type:
                    # Get simplified features for the potential antecedent
                    antecedent_features = {
                        'POS': antecedent_pos,
                        'NER': antecedent_ner,
                        'Lemma': antecedent_info['lemma'],
                        'Text': antecedent_info['text'],
                        'ExtractedGender': antecedent_info['extracted_gender'], # May be None
                        'ExtractedNumber': antecedent_info['extracted_number'], # May be None
                    }

                    # Check for agreement
                    if check_agreement(pronoun_reqs, antecedent_features):
                        # Found a matching antecedent - take the first one found (closest)
                        best_antecedent = antecedent_features
                        break # Stop searching backward for this pronoun

            if best_antecedent:
                # Replace the pronoun with the antecedent's text
                resolved_tokens.append(best_antecedent['Text'])
            else:
                # No suitable antecedent found, keep the original pronoun text
                resolved_tokens.append(token.text)
        else:
            # Not a target pronoun, keep the original token text
            resolved_tokens.append(token.text)

    # Join the tokens back into a single string.
    # Need to be careful with spaces and punctuation.
    # The simple join below won't handle spaces correctly between tokens.
    # A proper solution would involve preserving original spacing or using
    # the token start/stop offsets. For simplicity, we'll re-split the original
    # text and replace words at the correct positions.

    # --- Alternative: Build result based on original tokens and replace text ---
    output_tokens = []
    token_index = 0
    for original_token in re.split(r'(\s+|\W+)', text): # Split original text similarly
        if original_token.strip() == '': # Keep whitespace
            output_tokens.append(original_token)
            continue

        # Find the corresponding token in our processed list by text/lemma
        # This is fragile if words repeat
        matching_processed_token_index = -1
        for idx, info in enumerate(tokens_info[token_index:]):
             if info['text'].lower() == original_token.lower():
                  matching_processed_token_index = token_index + idx
                  break

        if matching_processed_token_index != -1:
             # If this token was replaced during resolution, use the replacement
             output_tokens.append(resolved_tokens[matching_processed_token_index])
             token_index = matching_processed_token_index + 1
        else:
             # If not found (e.g., punctuation handled by split but not in tokens_info)
             # or if the text matching failed, just keep the original token.
             # This part might need refinement for complex cases.
             output_tokens.append(original_token)


    return "".join(output_tokens)

def resolve_coreferences_english_with_spacy(text: str, nlp: spacy.Language = None) -> str:
    """
    Resolves coreferences in English text, replacing pronouns with their referents.
    For Russian, returns original text (coreference not implemented).
    """
    if nlp is None:
        return text

    doc = nlp(text)
    if not hasattr(doc, "coref_clusters"):
        logger.warning("Coreference resolution not available")
        return text

    resolved_text = list(doc._.coref_resolved) if doc._.coref_resolved else list(doc.text)
    result = ""
    for token, resolved in zip(doc, resolved_text):
        result += resolved if resolved else token.text
        if token.whitespace_:
            result += " "

    logger.debug(f"Coreference resolved: {text} -> {result}")
    return result


# --- Example Usage ---
if __name__ == '__main__':
    text1 = "Иван пошел в магазин. Он купил хлеб." # 'Он' -> 'Иван'
    text2 = "Мария читает книгу. Она очень интересная." # 'Она' -> 'Мария'
    text3 = "Студенты сдали экзамен. Они были довольны." # 'Они' -> 'Студенты'
    text4 = "Директор школы, Анна Петровна, выступила с речью. Ее слова вдохновили всех." # 'Ее' -> 'Анна Петровна' (Need to handle multi-word antecedents)
    text5 = "Кот сидел на окне. Его хвост свисал вниз." # 'Его' -> 'Кот'
    text6 = "Родители приехали вчера. Их приезд был сюрпризом." # 'Их' -> 'Родители'
    text_ambiguous = "Иван встретил Петра. Он улыбнулся." # 'Он' could be Ivan or Peter - rule picks last (Петр if added)
    text_multi_sentence = "Иван пошел домой. По дороге он зашел в пекарню. Там Иван купил хлеб. Хлеб был свежий. Он вкусно пах." # 'Он' -> 'Иван'

    # Add Петр to our antecedent feature extraction for the ambiguous case test
    # In a real system, NER would handle this better, but demonstrating feature extraction
    def get_antecedent_features_with_petr(token, ner_tag, doc_tokens):
        features = get_antecedent_features(token, ner_tag, doc_tokens)
        if token.lemma == 'петр': # Manual add for testing
            features['POS'] = token.pos # Should be PROPN
            features['NER'] = ner_tag # Should be PER
            features['Lemma'] = token.lemma
            features['Text'] = token.text
            # Assume Masculine Singular for Петр if morphology missing
            features['ExtractedGender'] = features.get('ExtractedGender') or 'Masc'
            features['ExtractedNumber'] = features.get('ExtractedNumber') or 'Sing'
        return features

    # Temporarily replace the function pointer for the ambiguous test
    original_get_antecedent_features = get_antecedent_features
    get_antecedent_features = get_antecedent_features_with_petr # Use modified version for test


    print(f"Original 1: {text1}")
    print(f"Resolved 1: {resolve_coreference_with_natasha_rules(text1)}")
    print("-" * 20)

    print(f"Original 2: {text2}")
    print(f"Resolved 2: {resolve_coreference_with_natasha_rules(text2)}")
    print("-" * 20)

    print(f"Original 3: {text3}")
    print(f"Resolved 3: {resolve_coreference_with_natasha_rules(text3)}")
    print("-" * 20)

    print(f"Original 4: {text4}") # Note: Multi-word antecedent "Анна Петровна" might be replaced by just "Анна" or "Петровна" depending on tokenization/NER span handling
    print(f"Resolved 4: {resolve_coreference_with_natasha_rules(text4)}")
    print("-" * 20)

    print(f"Original 5: {text5}")
    print(f"Resolved 5: {resolve_coreference_with_natasha_rules(text5)}")
    print("-" * 20)

    print(f"Original 6: {text6}")
    print(f"Resolved 6: {resolve_coreference_with_natasha_rules(text6)}")
    print("-" * 20)

    print(f"Original Ambiguous: {text_ambiguous}")
    print(f"Resolved Ambiguous: {resolve_coreference_with_natasha_rules(text_ambiguous)}") # Should resolve 'Он' to 'Петр' based on 'last matching' rule
    print("-" * 20)

    print(f"Original Multi-sentence: {text_multi_sentence}")
    print(f"Resolved Multi-sentence: {resolve_coreference_with_natasha_rules(text_multi_sentence)}") # Should resolve both 'Он' instances to 'Иван'
    print("-" * 20)

    # Restore original function pointer
    get_antecedent_features = original_get_antecedent_features