from common import *
import sys

vs_pc = False
if len(sys.argv) > 1:
    if "pc" in sys.argv:
        print("Computer opponent")
        vs_pc = True

def input_move(board):
    done = False
    while not done:
        uci = input()
        try:
            move = chess.Move.from_uci(uci)
            done = board.is_legal(move)
        except ValueError:
            pass
    return move

model = PolicyModel().to(get_device())
model.load_state_dict(torch.load("models/reinforce.pt",
    map_location=get_device()))

opp_model = PolicyModel().to(get_device())
opp_model.load_state_dict(torch.load("models/reinforce.pt",
    map_location=get_device()))

board = chess.Board()
for epoch in range(10000):
    my_side = epoch % 2 == 0
    print(board)
    print()
    while not board.is_game_over():
        if board.turn == my_side:
            move = choose_move(board, model, 0)
        else:
            if vs_pc:
                move = choose_move(board, opp_model, 0)
                input()
            else:
                move = input_move(board)
        board.push(move)
        print(board)
        print()

    board.reset()
