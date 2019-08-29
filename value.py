from common import *

def train(model, criterion, opt, board, meta, reward):
    model.zero_grad()
    pred = model(board, meta)
    loss = criterion(pred, reward)
    loss.backward()
    opt.step()
    return loss

batch_size = 4096

boards_all = torch.load("proc/value-net-boards.pt").type(
    torch.float).to(get_device())
meta_all = torch.load("proc/value-net-meta.pt").type(
    torch.float).to(get_device())
rewards_all = torch.load("proc/value-net-rewards.pt").type(
    torch.float).to(get_device())

model = ValueModel().to(get_device())
criterion = nn.MSELoss()
opt = optim.Adam(model.parameters())

for epoch in range(1000000):
    print("Epoch {}".format(epoch))
    running_loss = 0
    for batch_idx in range(0, boards_all.shape[0], batch_size):
        boards = boards_all[batch_idx:batch_idx+batch_size]
        meta = meta_all[batch_idx:batch_idx+batch_size]
        rewards = rewards_all[batch_idx:batch_idx+batch_size]
        loss = train(model, criterion, opt, boards, meta, rewards)
        running_loss += loss

    print("Loss: {:.6f}".format(running_loss.item()))

    torch.save(model.state_dict(), "models/value.pt")
