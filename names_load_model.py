from utils import LABELS_TO_TEXT, char_to_tensor, get_dict_len
from names_RNN import RNN
import torch as pt


model = RNN(get_dict_len(), 64, 2)
model.load_state_dict(pt.load("model.pth"))

model.eval()

print("Type a name:")
while True:
    inp = input(">").lower()

    hidden = model.init_hidden()

    for char in inp:
        output, hidden = model.forward(pt.reshape(
            char_to_tensor(char), (1, get_dict_len())), hidden)
    label = LABELS_TO_TEXT[output.argmax(1).item()]
    print(f"{inp} has the label: {label}")
