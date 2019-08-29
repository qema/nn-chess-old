from common import *

boards = torch.load("proc/boards-small.pt").type(torch.float).to(get_device())
meta = torch.load("proc/meta-small.pt").type(torch.float).to(get_device())
actions = torch.load("proc/actions-small.pt").to(get_device())
print("loaded proc")

model = PolicyModel().to(get_device())
criterion = nn.NLLLoss()
opt = optim.Adam(model.parameters())

for epoch in range(100000):
    print("Epoch {}".format(epoch))
    loss = train(model, criterion, opt, boards, meta, actions)
    print("Loss: {:.6f}".format(loss.item()))

    pred = model(boards, meta)
    print("Train acc: {:.4f}".format(
        (pred.argmax(dim=1) == actions).sum().item() / len(actions)))
    print()

    torch.save(model.state_dict(), "models/supervised.pt")

    #board = chess.Board(states[3])
    #b, meta = states_to_tensor([states[3]])
    #action_v = model(b, meta)
    #action = action_v.argmax().item()
    #print(board)
    #move = chess.Move(action // 64, action % 64)
    #if board.is_legal(move):
    #    board.push(move)
    #else:
    #    print("not legal:", move)
    #print(board)
    #print()
    #print()

