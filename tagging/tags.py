import csv
import pathlib
import pickle
import re
from collections import defaultdict
from typing import List, Dict
from tqdm import tqdm

import nltk.tokenize
import numpy as np

_DEFINITIONS_FILE = './definitions.csv'
_DEFAULT_CACHE_DIR = f'{str(pathlib.Path.home())}/ImageD/tagging'

_common_tags = None
_mask_size = None


def _tokenize(definition: str) -> List[str]:
    low = definition.lower()
    tokens = nltk.tokenize.word_tokenize(low)
    words = [token for token in tokens if re.fullmatch(r"[a-z]+", token) is not None]
    return words


def _split_definitions():
    with open(_DEFINITIONS_FILE) as file:
        reader = csv.reader(file)
        next(reader)
        definitions = [row for row in reader]
    return [[definition[0], _tokenize(definition[1])] for definition in definitions]


def _extract_tags(tokenized_definitions) -> np.ndarray:
    frequency_per_definition = defaultdict(lambda: 0)
    for definition in tokenized_definitions:
        unique_words = set(definition[1])
        for word in unique_words:
            frequency_per_definition[word] += 1

    definitions_count = len(tokenized_definitions)
    threshold = definitions_count / 100
    tags = [word for word, freq in frequency_per_definition.items() if freq < threshold]
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


def _prepare_tags() -> Dict[str, np.ndarray]:
    print('Preparing tags from definitions...')
    tokenized_definitions = _split_definitions()
    tags = _extract_tags(tokenized_definitions)
    result = _digitalize_tags(tags, tokenized_definitions)
    print('Done.')
    return result


def attach_tags(dataset: np.ndarray, label='train-split', cache_dir=_DEFAULT_CACHE_DIR) -> Dict[str, np.ndarray]:
    cache_path_raw = f'{cache_dir}/{label}.bin'
    cache_path = pathlib.Path(cache_path_raw)
    if cache_path.exists():
        with open(cache_path, 'rb') as cache_file:
            try:
                print('Loading tags from cache.')
                return pickle.load(cache_file)
            except pickle.UnpicklingError as e:
                print('Failed to load tags, restoring manually.')

    global _common_tags
    if _common_tags is None:
        _common_tags = _prepare_tags()

    mask_dimension = len(next(iter(_common_tags.values())))
    records = defaultdict(lambda: np.zeros(mask_dimension))
    for image_record in tqdm(dataset, 'Attaching tags to dataset'):
        image_id = image_record[0]
        label = image_record[1]
        mask = _common_tags[label]
        records[image_id] = np.logical_or(records[image_id], mask)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path_raw, 'wb+') as cache_file:
        pickle.dump(dict(records), cache_file)
    return records


def tags() -> Dict[str, np.ndarray]:
    global _common_tags
    if _common_tags is None:
        _common_tags = _prepare_tags()
    return _common_tags


def tag_mask_size() -> int:
    global _mask_size
    if _mask_size is None:
        t = tags()
        first = next(iter(t.values()))
        _mask_size = first.shape[0]
    return _mask_size
