import random
from common import *
import torch.multiprocessing as mp

game_batch_size = 10
max_recent_opps = 10000
reward_dict = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}

def train(model, opt, criterion, boards, metas, actions, reward):
    model.zero_grad()
    pred = model(boards, metas)
    loss = criterion(pred, actions)
    loss *= reward
    loss = torch.mean(loss)
    loss.backward()
    opt.step()
    return loss

def run_game(model, opp_model, epoch):
    moves, states, rewards = [], [], []
    board = chess.Board()
    my_side = epoch % 2 == 0
    n_moves = 0
    while not board.is_game_over():
        if board.turn == my_side:
            move = choose_move(board, model, 0)
            # record for replay
            moves.append(move.uci())
            states.append(board.fen())
            n_moves += 1
        else:
            move = choose_move(board, opp_model, 0)
        board.push(move)

    result = board.result()
    reward = reward_dict[result]
    if not my_side:
        reward *= -1
    rewards += [reward]*n_moves

    return moves, states, rewards

if __name__ == "__main__":
    model = PolicyModel().to(get_device())
    model.load_state_dict(torch.load("models/supervised.pt"))

    opp_model = PolicyModel()
    opp_model.load_state_dict(torch.load("models/supervised.pt"))
    opp_model_pool = []

    opt = optim.Adam(model.parameters())
    criterion = nn.NLLLoss(reduction="none")

    for epoch in range(10000):
        print("Epoch {}".format(epoch))
        # play n games
        with torch.no_grad():
            moves, states, rewards = [], [], []
            for n in range(game_batch_size):
                m, s, r = run_game(model, opp_model, epoch)
                moves += m
                states += s
                rewards += r

        # train
        boards, metas = states_to_tensor(states)
        boards = boards.type(torch.float)
        metas = metas.type(torch.float)
        actions = actions_to_tensor(moves)
        rewards = torch.tensor(rewards, dtype=torch.float)
        loss = train(model, opt, criterion, boards, metas, actions, rewards)
        print("Loss: {:.6f}".format(loss.item()))
        print()

        torch.save(model.state_dict(), "models/reinforce.pt")

        if epoch % 2 == 0:
            opp_model_pool.append(model.state_dict())
            opp_model_pool = opp_model_pool[-max_recent_opps:]
            params = random.choice(opp_model_pool)
            opp_model.load_state_dict(params)
