import re
import string
from collections import Counter
from typing import Pattern, TypeAlias


PosLexicon: TypeAlias = dict[str, list[str]]
WordLookup: TypeAlias = dict[str, str]
WordWeights: TypeAlias = tuple[list[str], list[float]]
TemplateFieldWordWeights: TypeAlias = dict[str, dict[str, WordWeights]]
TemplateFieldCounts: TypeAlias = dict[str, dict[str, Counter[str]]]


def normalize_token(token: str) -> str:
    normalized = token.strip().lower()
    normalized = normalized.strip(string.punctuation + "“”‘’")
    return normalized


def clean_slot_word(word: str) -> str:
    return normalize_token(word)


def is_valid_slot_word(
    word: str,
    slot_word_regex: Pattern[str],
    min_slot_word_length: int,
    max_slot_word_length: int,
) -> bool:
    if (len(word) < min_slot_word_length) or (len(word) > max_slot_word_length):
        return False

    return slot_word_regex.match(word) is not None


def load_part_of_speech_words(
    part_of_speech_filename: str,
    slot_word_regex: Pattern[str],
    min_slot_word_length: int,
    max_slot_word_length: int,
) -> PosLexicon:
    words = {}

    with open(part_of_speech_filename, "r", encoding="utf-8") as file_obj:
        lines = [line.strip() for line in file_obj if line.strip()]

    current_pos = None
    for line in lines:
        if line.startswith("*") and line.endswith("*"):
            current_pos = line.strip("*")
            words[current_pos] = []
            continue

        row_words = [word.strip() for word in line.split(",")]
        cleaned_words = [clean_slot_word(word) for word in row_words]
        filtered_words = [
            word
            for word in cleaned_words
            if is_valid_slot_word(
                word, slot_word_regex, min_slot_word_length, max_slot_word_length
            )
        ]
        words[current_pos].extend(filtered_words)

    return words


def build_word_to_pos_lookup(words: PosLexicon) -> WordLookup:
    lookup = {}
    for pos_name, pos_words in words.items():
        for word in pos_words:
            normalized_word = normalize_token(word)
            if normalized_word and normalized_word not in lookup:
                lookup[normalized_word] = pos_name

    return lookup


def tokenize_with_regex(text: str, token_regex: Pattern[str]) -> list[str]:
    return token_regex.findall(text)


def strip_numeric_suffix(field_name: str) -> str:
    return re.sub(r"[0-9]+$", "", field_name)


def template_fields_in_order(template: str) -> list[str]:
    fields = []
    formatter = string.Formatter()

    for _, field_name, _, _ in formatter.parse(template):
        if field_name and field_name not in fields:
            fields.append(field_name)

    return fields


def finalize_template_field_word_weights(
    template_field_counts: TemplateFieldCounts,
    selected_templates: list[str],
    min_count: int,
) -> TemplateFieldWordWeights:
    result = {}
    for template in selected_templates:
        if template not in template_field_counts:
            continue

        field_weights = {}
        for field_name, counts in template_field_counts[template].items():
            filtered_items = [
                (word, count) for word, count in counts.items() if count >= min_count
            ]
            if not filtered_items:
                filtered_items = list(counts.items())
            field_words = [word for word, _ in filtered_items]
            weight_values = [float(count) for _, count in filtered_items]
            field_weights[field_name] = (field_words, weight_values)

        if field_weights:
            result[template] = field_weights

    return result
