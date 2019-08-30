from common import *
import sys
import numpy as np

model = PolicyModel().to(get_device())
model.load_state_dict(torch.load("models/reinforce.pt",
    map_location=get_device()))

opp_model = PolicyModel().to(get_device())
opp_model.load_state_dict(torch.load("models/supervised.pt",
    map_location=get_device()))

rewards = []
board = chess.Board()
for epoch in range(100):
    my_side = epoch % 2 == 0
    #print(board)
    #print()
    while not board.is_game_over():
        if board.turn == my_side:
            move = choose_move(board, model, 0)
        else:
            move = choose_move(board, opp_model, 0)
        board.push(move)
        #print(board)
        #print()

    reward = reward_for_side(board, my_side)
    rewards.append(reward)
    print("Game {}. Reward: {}".format(epoch, reward))

    board.reset()

print("Average reward for RL: {:.6f}".format(np.mean(rewards)))
