# %%
from names_RNN import RNN
from typing import List
import torch
from torch.functional import Tensor
from names_dataset import NamesDataset
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from utils import get_dict_len
import matplotlib.pyplot as plt
# %%

training_data = NamesDataset()
test_data = NamesDataset(train=False)

# %%

train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)


model = RNN(get_dict_len(), 128, 2)
# %%

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# %%


def train(input: List[Tensor], label):
    hidden = model.init_hidden()

    model.zero_grad()

    for tensor in input:
        output, hidden = model.forward(input=tensor, hidden=hidden)

    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

    return output, loss.item()


# %%

print("Training model")

epochs = 5

losses = []

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    sum_loss = 0
    for idx, data in enumerate(train_dataloader):
        X, y = data
        output, loss = train(X, y)
        sum_loss += loss
        if (idx + 1) % 1000 == 0:
            print(sum_loss/1000)
            losses.append(sum_loss/1000)
            sum_loss = 0

# %%

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# %%

print("Testing")

size = len(test_dataloader.dataset)
model.eval()
test_loss, correct = 0, 0
with torch.no_grad():
    for data in test_dataloader:
        X, y = data
        hidden = model.init_hidden()

        for tensor in X:
            pred, hidden = model.forward(input=tensor, hidden=hidden)

        test_loss += criterion(pred, y.long()).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

test_loss /= size
correct /= size
print(
    f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
# %%
plt.plot(losses)
plt.show()

# %%
