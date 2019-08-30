from common import *
import numpy as np
from collections import namedtuple

class GameTreeNode:
    def __init__(self):
        self.ps = np.array([])
        self.nvs = np.array([])
        self.nrs = np.array([])
        self.wvs = np.array([])
        self.wrs = np.array([])
        self.moves = []  # uci
        self.move_to_idx = {}
        self.children = []
        self.actives = []

    def add_child(self, move, p, nv, nr, wv, wr):
        self.move_to_idx[move] = len(self.moves)
        self.moves.append(move)
        self.ps = np.append(self.ps, p)
        self.nvs = np.append(self.nvs, nv)
        self.nrs = np.append(self.nrs, nr)
        self.wvs = np.append(self.wvs, wv)
        self.wrs = np.append(self.wrs, wr)
        self.actives.append(False)
        self.children.append(GameTreeNode())

class MCTS:
    def __init__(self, board, tree_policy, rollout_policy, value_model,
        lamb, c_puct, n_thr):
        self.tree_policy = tree_policy
        self.rollout_policy = rollout_policy
        self.value_model = value_model
        self.lamb = lamb
        self.c_puct = c_puct
        self.n_thr = n_thr

        self.board = chess.Board(board.fen())
        self.game_tree_root = GameTreeNode()
        for move in self.board.legal_moves:
            self.game_tree_root.add_child(move.uci(), 0, 0, 0, 0, 0)

    # returns: board state at leaf, leaf node, path of indices to leaf
    def select_leaf(self):
        board = chess.Board(self.board.fen())
        cur = self.game_tree_root
        path = []
        eps = 1e-7
        while cur.children:
            q = ((1-self.lamb)*cur.wvs/(cur.nvs+eps) +
                self.lamb*cur.wrs/(cur.nrs+eps))
            num = np.sqrt(np.sum(cur.nrs))
            denom = 1 + cur.nrs
            u = self.c_puct * cur.ps * num/denom

            scores = q + u

            idx = np.argmax(scores)
            path.append(idx)
            board.push(chess.Move.from_uci(cur.moves[idx]))
            cur = cur.children[idx]
        #print(len(self.game_tree_root.children), path)
        return board, cur, path

    # NB: modifies board
    def eval_node(self, board, node):
        my_side = self.board.turn

        # value net
        board_t, meta_t = states_to_tensor([board.fen()])
        board_t = board_t.type(torch.float)
        meta_t = meta_t.type(torch.float)
        pred_value = self.value_model(board_t, meta_t)[0].item()

        # rollout
        while not board.is_game_over():
            move = choose_move(board, self.rollout_policy, 0)
            board.push(move)            
        rollout_value = reward_for_side(board, my_side)

        return pred_value, rollout_value

    # board_fen: fen of board right before first rollout move
    def backup(self, path, pred_value, rollout_value, board_fen):
        def update_node(node, idx):
            node.nrs[idx] += 1
            node.nvs[idx] += 1
            node.wrs[idx] += rollout_value
            node.wvs[idx] += pred_value
        cur = self.game_tree_root
        for idx in path:
            update_node(cur, idx)

            # if past thresh, initialize new node's prior prob and
            # create its children
            if cur.nrs[idx] >= self.n_thr and not cur.actives[idx]:
                #print("new")
                cur.actives[idx] = True
                board_t, meta_t = states_to_tensor([board_fen])
                board_t = board_t.type(torch.float)
                meta_t = meta_t.type(torch.float)
                action_t = actions_to_tensor([cur.moves[idx]])[0]
                p = self.tree_policy(board_t, meta_t)[0][action_t].item()
                cur.ps[idx] = p
                
                board = chess.Board(board_fen)
                for move in board.legal_moves:
                    cur.children[idx].add_child(move.uci(), 0, 0, 0, 0, 0)

            cur = cur.children[idx]

    def search_step(self):
        board, node, path = self.select_leaf()
        board_fen = board.fen()
        pred_value, rollout_value = self.eval_node(board, node)
        self.backup(path, pred_value, rollout_value, board_fen)

    # returns best move (uci)
    def search(self, n_steps):
        for i in range(n_steps):
            self.search_step()

        root = self.game_tree_root
        best_move_idx = np.argmax(root.nvs + root.nrs)
        return root.moves[best_move_idx]

    # commit move based on move uci
    def commit_move_uci(self, move_uci):
        move_idx = self.game_tree_root.move_to_idx[move_uci]
        self.game_tree_root = self.game_tree_root.children[move_idx]
        self.board.push(chess.Move.from_uci(move_uci))
        for move in self.board.legal_moves:
            uci = move.uci()
            if uci not in self.game_tree_root.move_to_idx:
                self.game_tree_root.add_child(uci, 0, 0, 0, 0, 0)

if __name__ == "__main__":
    tree_policy = PolicyModel().to(get_device())
    tree_policy.load_state_dict(torch.load("models/supervised.pt",
        map_location=get_device()))
    rollout_policy = PolicyModel().to(get_device())
    rollout_policy.load_state_dict(torch.load("models/reinforce.pt",
        map_location=get_device()))
    value_model = ValueModel().to(get_device())
    value_model.load_state_dict(torch.load("models/value.pt",
        map_location=get_device()))

    board = chess.Board()

    searcher = MCTS(board, tree_policy, rollout_policy, value_model, 1, 1, 10)
    #searcher.game_tree_root.add_child("a2a3", 0.3, 0, 0, 2, 2)
    #searcher.game_tree_root.add_child("b2b3", 0.5, 0, 0, 1, 3)
    best_move = searcher.search(10)
    print(best_move)
