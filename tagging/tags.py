import re
from collections import defaultdict
from typing import List, Dict

import nltk.tokenize
import numpy as np

_DEFINITIONS_FILE = './definitions.csv'

_common_tags = None


def _tokenize(definition: str) -> List[str]:
    low = definition.lower()
    tokens = nltk.tokenize.word_tokenize(low)
    words = [token for token in tokens if re.fullmatch(r"[a-z]+", token) is not None]
    return words


def _split_definitions():
    definitions = np.recfromcsv(_DEFINITIONS_FILE, skip_header=1)
    return [[definition[0], _tokenize(definition[1])] for definition in definitions]


def _extract_tags(tokenized_definitions) -> np.ndarray:
    frequency_per_definition = defaultdict(lambda x: 0)
    for definition in tokenized_definitions:
        unique_words = set(definition[1])
        for word in unique_words:
            frequency_per_definition[word] += 1

    definitions_count = len(tokenized_definitions)
    threshold = definitions_count / 10
    tags = [word for word, freq in frequency_per_definition if freq < threshold]
    tags.sort()
    return tags


def _digitalize_tags(tags: List[str], definitions) -> Dict[str, np.ndarray]:
    result = dict()
    for definition in definitions:
        mask = np.zeros(len(tags))
        for i, tag in enumerate(tags):
            if tag in definition[1]:
                mask[i] = 1
        result[definition[0]] = mask
    return result


def _prepare_tags():
    tokenized_definitions = _split_definitions()
    tags = _extract_tags(tokenized_definitions)
    global _common_tags
    return _digitalize_tags(tags, tokenized_definitions)


def attach_tags(dataset: np.ndarray) -> Dict[str, np.ndarray]:
    global _common_tags
    if _common_tags is None:
        _common_tags = _prepare_tags()
    records = defaultdict(lambda x: np.zeros(len(_common_tags)))
    for image_record in dataset:
        image_id = image_record[0]
        label = image_record[1]
        mask = _common_tags[label]
        records[image_id] = np.logical_or(records[image_id], mask)
    return records
