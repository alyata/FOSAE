import numpy as np

def generate_example():
    a = np.array(range(9), dtype=np.int8)
    np.random.shuffle(a)
    a = a.reshape(3, 3)
    example = []
    for row in range(3):
        for col in range(3):
            e = np.zeros(15)
            index = a[row, col]
            e[index] = 1
            e[9 + row] = 1
            e[12 + col] = 1
            example.append(e)
    example = np.array(example)
    np.random.shuffle(example)
    return example

train_set = np.array([generate_example() for _ in range(18000)])
test_set = np.array([generate_example() for _ in range(2000)])

np.save("train_set.npy", train_set)
np.save("test_set.npy", test_set)
