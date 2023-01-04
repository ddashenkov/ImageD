import sys
from collections import defaultdict

lines_per_chunk = 3000000

chunks = defaultdict(list)

file_name = sys.argv[1]
with open(file_name, 'r') as file:
    header = file.readline()
    for i, line in enumerate(file):
        chunk_index = i // lines_per_chunk
        chunks[chunk_index].append(line.strip())

        if i % lines_per_chunk == 0:
            print(f'Chunk {chunk_index}')

file_name_without_extension = file_name.strip('.csv')
for index, lines in chunks.items():
    with open(file_name_without_extension + '.chunk' + str(index) + '.csv', 'w') as file:
        content = header + '\n'.join(lines)
        file.write(content)
