from common import *
from mcts import *

tree_policy = PolicyModel().to(get_device())
tree_policy.load_state_dict(torch.load("models/supervised.pt",
    map_location=get_device()))
rollout_policy = PolicyModel().to(get_device())
rollout_policy.load_state_dict(torch.load("models/reinforce.pt",
    map_location=get_device()))
value_model = ValueModel().to(get_device())
value_model.load_state_dict(torch.load("models/value.pt",
    map_location=get_device()))

with torch.no_grad():
    while True:
        board = chess.Board()
        w_searcher = MCTS(board, True,
            tree_policy, rollout_policy, value_model,
            1, 1, 0.1, 10)
        b_searcher = MCTS(board, False,
            tree_policy, rollout_policy, value_model,
            1, 1, 0.1, 10)
        while not board.is_game_over():
            print(board)
            print()
            #searcher.game_tree_root.add_child("a2a3", 0.3, 0, 0, 2, 2)
            #searcher.game_tree_root.add_child("b2b3", 0.5, 0, 0, 1, 3)
            if board.turn:
                move = w_searcher.search(100)
            else:
                move = b_searcher.search(100)
            board.push(chess.Move.from_uci(move))
            w_searcher.commit_move_uci(move)
            b_searcher.commit_move_uci(move)

            print(move)
        print(board)
