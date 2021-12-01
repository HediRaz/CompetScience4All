import graph.load_data
import graph.data_utils
import numpy as np

def create_testing_dataset(folder, proportion, minimal_degs, size):
    dataset = graph.load_data.create_dataset_from_to(2014, 2017, proportion, minimal_degs, size)
    pairs = np.array(dataset[1], dtype=np.int64)
    labels = np.array(dataset[2], dtype=np.int64)

    np.save(folder + "pairs.npy", pairs)
    np.save(folder + "labels.npy", labels)

if __name__ == "__main__":
    create_testing_dataset("./test_dataset/", 1_000, 5, 1_000_000//1_000)
