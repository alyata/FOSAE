import numpy as np
from torch import autograd
import torch
from FOSAE import FOSAE, pairwiseSquareLoss

epoch = 10
learn_rate = 1e-3

train = np.load("train_set.npy")
test = np.load("test_set.npy")
model = FOSAE(9, 15, 2, 25, 10)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)
loss = pairwiseSquareLoss
#loss = torch.nn.MSELoss()
#loss = torch.nn.BCELoss()

def reconstruct_grid(input):
    grid = torch.tensor(
        [[-1, -1, -1],
         [-1, -1, -1],
         [-1, -1, -1]]
    )
    for tile in input:
        digit = torch.argmax(tile[0:9])
        row = torch.argmax(tile[9:12])
        col = torch.argmax(tile[12:15])
        grid[row, col] = digit
    return grid

for epoch_num in range(epoch):
    n = 0
    total_loss = 0
    for example in train:
        example = torch.tensor(example).float()
        output = model(example)
        l = loss(output, example)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        total_loss += l.item()
        n += 1
        if n % 100 == 0:
            print(f"iter {n} avg loss: {total_loss/100}")
            total_loss = 0

    total_test_loss = 0
    n = 0
    for example in test:
        example = torch.tensor(example).float()
        output = model(example)
        l = loss(output, example)
        total_test_loss += l.item()
        n+= 1
        if n % 100 == 0:
            print("expected:")
            print(reconstruct_grid(example))
            print("actual:")
            print(reconstruct_grid(output))
            print(output)
            print("------------------------------")
    print(f"epoch {epoch_num} test loss: {total_test_loss/len(test)}")
