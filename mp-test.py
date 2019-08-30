import torch
from common import *
import torch.nn as nn
import torch.multiprocessing as mp

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()

    def forward(self, x):
        return x * x

def f(idx, model, x, q):
    b = torch.zeros(1, 12, 8, 8)
    m = torch.zeros(1, 8)
    print("before")
    model(b, m)
    q.put("hello")

if __name__ == "__main__":
    with torch.no_grad():
        model = PolicyModel().to(get_device())
        model.share_memory()
        model.load_state_dict(torch.load("models/supervised.pt"))

        q = mp.Queue()
        x = torch.tensor([3])
        p = mp.Process(target=f, args=(0, model, x, q))
        p.start()
        print(q.get())
        p.join()
