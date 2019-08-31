import chess
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import random

reward_dict = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
# precond: game is over
def reward_for_side(board, side):
    result = board.result()
    reward = reward_dict[result]
    if not side:
        reward *= -1
    return reward

class ValueModel(nn.Module):
    def __init__(self):
        super(ValueModel, self).__init__()
        self.conv1 = nn.Conv2d(12, 20, 2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(20, 20, 2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(20, 20, 2)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(508, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 512)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(512, 1)

    def forward(self, board, meta):
        out = self.conv1(board)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.shape[0], -1)
        out = torch.cat((out, meta), dim=1)
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.fc2(out)
        out = self.relu5(out)
        out = self.fc3(out)
        return out

class PolicyModel(nn.Module):
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.conv1 = nn.Conv2d(12, 128, 2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 128, 2)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(3208, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(256, 64*64)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, board, meta):
        out = self.conv1(board)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.shape[0], -1)
        out = torch.cat((out, meta), dim=1)
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.fc2(out)
        out = self.relu5(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out

device_cache = None
def get_device():
    global device_cache
    if device_cache is None:
        device_cache = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
    return device_cache

def load_data(cap=None):
    winners = []
    games = []
    with open("games.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        winners_idx = header.index("winner")
        moves_idx = header.index("moves")
        for row in reader:
            winners.append(row[winners_idx])
            games.append(row[moves_idx].split(" "))

    states, actions = [], []
    print(len(games))
    for i, game in enumerate(games):
        if i % 1000 == 0: print(i)
        if cap is not None and i == cap: break
        board = chess.Board()
        for move_san in game:
            move = board.parse_san(move_san)
            states.append(board.fen())
            actions.append(move.uci())
            board.push(move)
    return states, actions

def states_to_tensor(states, verbose=False):
    board_tensors, meta_tensors = [], []
    board = chess.Board()
    if verbose: print(len(states))
    for t, state in enumerate(states):
        if verbose and t % 1000 == 0: print(t)
        board.set_fen(state)
        piece_map = board.piece_map()
        board_t = torch.zeros(12, 8, 8, dtype=torch.uint8, device=get_device())
        piece_to_idx = {p: i for i, p in enumerate("pbnrqkPBNRQK")}
        for pos, piece in piece_map.items():
            col, row = chess.square_file(pos), chess.square_rank(pos)
            board_t[piece_to_idx[piece.symbol()]][row][col] = 1

        meta_tensors.append(torch.tensor([
            board.turn,
            board.has_kingside_castling_rights(True),
            board.has_kingside_castling_rights(False),
            board.has_queenside_castling_rights(True),
            board.has_queenside_castling_rights(False),
            board.has_legal_en_passant(),
            board.halfmove_clock,
            board.fullmove_number
        ], dtype=torch.long, device=get_device()))

        board_tensors.append(board_t)
    board_tensors = torch.stack(board_tensors)
    meta_tensors = torch.stack(meta_tensors)
    return board_tensors, meta_tensors

def action_idx_to_move(idx):
    from_to = idx# // 6
    #promot = idx % 6
    #if promot == 0:
    #    promot = None
    return chess.Move(from_to // 64, from_to % 64)#, promot)

def move_to_action_idx(move):
    return move.from_square*64 + move.to_square

def actions_to_tensor(actions, verbose=False):
    action_tensors = []
    for t, action in enumerate(actions):
        if verbose and t % 1000 == 0: print(t)
        move = chess.Move.from_uci(action)

        if move.promotion is None:
            move.promotion = 0

        action_tensors.append(move_to_action_idx(move))

        #action_t = torch.zeros(64, 64)
        #action_t[move.from_square][move.to_square] = 1
        #action_tensors.append(action_t.view(-1, 64*64))
    action_tensors = torch.tensor(action_tensors, dtype=torch.long,
        device=get_device())
    return action_tensors

def choose_move(board, model, eps):
    legal_moves = list(board.legal_moves)
    if random.random() < eps:
        move = random.choice(legal_moves)
    else:
        board_t, meta_t = states_to_tensor([board.fen()])
        board_t = board_t.type(torch.float)
        meta_t = meta_t.type(torch.float)
        pred = model(board_t, meta_t)

        valid_idxs = [move_to_action_idx(move) for move in legal_moves]
        pred = pred[0][valid_idxs]
        actions = torch.distributions.Categorical(logits=pred)
        move = legal_moves[actions.sample().item()]
        if move.promotion is not None:
            move.promotion = 5
    return move

if __name__ == "__main__":
    # test stuff only
    states, actions = load_data(cap=100)
    boards_t, metas_t = states_to_tensor(states)
    actions_t = actions_to_tensor(actions)
    boards_t = boards_t.type(torch.float)
    metas_t = metas_t.type(torch.float)

    model = PolicyModel()
    criterion = nn.NLLLoss()
    opt = optim.Adam(model.parameters())

    for epoch in range(1000):
        print(epoch)
        loss = train(model, criterion, opt, boards_t, metas_t, actions_t)
        print(loss.item())

        #pred = model(boards_t, metas_t)
        #print((pred.argmax(dim=1) == actions_t).sum().item(), len(actions_t))

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
