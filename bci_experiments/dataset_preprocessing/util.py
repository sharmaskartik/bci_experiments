import pickle


def read_partition(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return data[0], data[1], data[2]


