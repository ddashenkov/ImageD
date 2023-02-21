import numpy as np
import csv

_CHUNKS_OF_TRAIN = 3


def read_train() -> np.ndarray:
    """
    Reads all the records of the train split.

    Records are stored in several files for technical reasons. Thus, this function joins the file contents and produces
    a Numpy array with all the data in the train split.

    The array contains has the shape `(N, 2)`, where `N` is the number of elements in the train split. The three columns
    are the unique ID of the image (string), the name of the label (string).

    :return: an `ndarray` with the train data
    """
    data = []
    for i in range(_CHUNKS_OF_TRAIN):
        file_name = f"train-annotation.chunk{i}.csv"
        path = "./" + file_name
        with open(path) as file:
            table = csv.reader(file)
            next(table)
            for row in table:
                data.append([row[0], row[1]])
    return np.asarray(data)

