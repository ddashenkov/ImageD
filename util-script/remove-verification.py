import sys

lines = []
with open(sys.argv[1], 'r') as file:
    for line in file:
        if len(lines) == 0:
            lines.append(line)
        else:
            parts = line.strip().split(',')
            if parts[3] == '1':
                del parts[1]
                lines.append(','.join(parts))

content = '\n'.join(lines)
with open(sys.argv[2], 'w') as file:
    file.write(content)
