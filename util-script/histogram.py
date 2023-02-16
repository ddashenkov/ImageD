import csv
import math
import sys

import matplotlib.pyplot as plt

frequencies = dict()

file_names = sys.argv[1:]
for file_name in file_names:
    with open(file_name, 'r') as file:
        csv_file = csv.reader(file)
        next(csv_file)
        for line in csv_file:
            label = line[1]
            if label not in frequencies:
                frequencies[label] = 0
            frequencies[label] += 1

freqs = sorted(frequencies.values(), reverse=True)
# plt.bar(range(len(freqs)), columns)
bin_count = 100
plt.xscale('log')
i, g, patches = plt.hist(freqs, bins=[2 ** x for x in range(0, 25)])

color = (0.5, 0.5, 1, 1)
other_color = (0.5, 1, 0.5, 1)
for patch in patches:
    patch.set_facecolor(color)

    t = color
    color = other_color
    other_color = t

plt.show()
