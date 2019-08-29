from common import *
import sys

batch_size = 2048
use_small = False
if len(sys.argv) > 1:
    if sys.argv[1] == "small":
        print("Using small data")
        use_small = True

s = "-small" if use_small else ""
boards_all = torch.load("proc/boards{}.pt".format(s)).type(
    torch.float).to(get_device())
meta_all = torch.load("proc/meta{}.pt".format(s)).type(
    torch.float).to(get_device())
actions_all = torch.load("proc/actions{}.pt".format(s)).to(get_device())
print("loaded proc")

model = PolicyModel().to(get_device())
criterion = nn.NLLLoss()
opt = optim.Adam(model.parameters())

for epoch in range(1000000):
    print("Epoch {}".format(epoch))
    running_loss = 0
    for batch_idx in range(0, boards_all.shape[0], batch_size):
        boards = boards_all[batch_idx:batch_idx+batch_size]
        meta = meta_all[batch_idx:batch_idx+batch_size]
        actions = actions_all[batch_idx:batch_idx+batch_size]
        loss = train(model, criterion, opt, boards, meta, actions)
        running_loss += loss

    print("Loss: {:.6f}".format(running_loss.item()))

    if epoch % 100 == 0:
        pred = model(boards, meta)
        print("Train acc: {:.4f}".format(
            (pred.argmax(dim=1) == actions).sum().item() / len(actions)))
        print()

    torch.save(model.state_dict(), "models/supervised.pt")
