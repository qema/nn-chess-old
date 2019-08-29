from common import *

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
model.load_state_dict(torch.load("models/reinforce.pt"))

board = chess.Board()
for epoch in range(10000):
    my_side = epoch % 2 == 0
    print(board)
    print()
    while not board.is_game_over():
        if board.turn == my_side:
            move = choose_move(board, model, 0)
        else:
            move = input_move(board)
        board.push(move)
        print(board)
        print()

    board.reset()
