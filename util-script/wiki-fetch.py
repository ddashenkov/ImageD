import csv
import sys

import nltk
import wikipedia
import tqdm

source_file_name = sys.argv[1]
target_file_name = sys.argv[2]

definitions = []
punctuation = {',', '.', '-', '—', ';', ':', "'", '"', '\n', '\t'}


def prepare_definition(wiki_article: str) -> str:
    tokens = nltk.tokenize.word_tokenize(wiki_article)
    words = [t for t in tokens if t not in punctuation]
    if len(words) > 512:
        words = words[:512]
    return ' '.join(words)


print(f'Reading terms from `{source_file_name}`.')
with open(source_file_name) as file:
    reader = csv.reader(file)
    next(reader)
    spreadsheet = [row for row in reader]

errors = []
ERROR_NOT_FOUND = 0
ERROR_OTHER = 1
ERROR_NETWORK = 2

for row in tqdm.tqdm(spreadsheet, 'Fetching Wiki articles...', leave=True):
    identifier = row[0]
    term = row[1]
    try:
        wiki_article = wikipedia.page(term)
        article = wiki_article.content
        definitions.append([identifier, prepare_definition(article)])
    except wikipedia.PageError as e:
        print(f'No Wiki page found for term {identifier}({term}).')
        print('\n')
        errors.append([identifier, term, ERROR_NOT_FOUND])
    except wikipedia.WikipediaException as e:
        print(f'Failed to fetch a Wikipedia article for {identifier}({term}): ' + str(e))
        print('\n')
        errors.append([identifier, term, ERROR_OTHER])
    except Exception as e:
        print(f'Failed to fetch a Wikipedia article for {identifier}({term}): ' + str(e))
        print('\n')
        errors.append([identifier, term, ERROR_NETWORK])

print(f'Writing definitions into `{target_file_name}`')
with open(target_file_name, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['LabelName', 'Definition'])
    writer.writerows(definitions)

print(f'Writing errors into `{target_file_name}.error.csv`')
with open(target_file_name + '.error.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['LabelName', 'DisplayName', 'ErrorType'])
    writer.writerows(errors)

print('Done.')
