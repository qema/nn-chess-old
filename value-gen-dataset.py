from common import *

dataset_size = 1000

s_model = PolicyModel().to(get_device())
s_model.load_state_dict(torch.load("models/supervised.pt",
    map_location=get_device()))

rl_model = PolicyModel().to(get_device())
rl_model.load_state_dict(torch.load("models/reinforce.pt",
    map_location=get_device()))

fens, rewards = [], []
for game_num in range(dataset_size):
    print("{} of {}".format(game_num, dataset_size))
    with torch.no_grad():
        success = False
        while not success:
            board = chess.Board()
            t = 0
            U = random.randint(0, 200)
            while not board.is_game_over():
                if t == U + 1:
                    side = board.turn
                    fen = board.fen()
                with torch.no_grad():
                    if t < U:      # pick move according to supervised model
                        move = choose_move(board, s_model, 0)
                    elif t == U:   # pick move uniformly at random
                        move = choose_move(board, s_model, 1)
                    else:          # pick move according to RL model
                        move = choose_move(board, rl_model, 0)

                board.push(move)
                t += 1

            if t > U + 1:
                fens.append(fen)
                rewards.append(reward_for_side(board, side))
                success = True

boards, meta = states_to_tensor(fens)
rewards = torch.tensor(rewards, dtype=torch.uint8)
torch.save(boards, "proc/value-net-boards.pt")
torch.save(meta, "proc/value-net-meta.pt")
torch.save(rewards, "proc/value-net-rewards.pt")
