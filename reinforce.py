import random
from common import *
import torch.multiprocessing as mp
import sys
import queue

n_workers = 4
game_batch_size = 128
max_recent_opps = 10000
pool_update_dur = 64

def train(model, opt, criterion, boards, metas, actions, reward):
    model.zero_grad()
    pred = model(boards, metas)
    loss = criterion(pred, actions)
    loss *= reward
    loss = torch.sum(loss) / game_batch_size
    loss.backward()
    opt.step()
    return loss

def run_game(n_games, queue, model, opp_model, epoch):
    moves = [[] for i in range(n_games)]
    states = [[] for i in range(n_games)]
    rewards = [[] for i in range(n_games)]
    boards = [chess.Board() for i in range(n_games)]
    n_done = 0
    t = 0

    while n_done < n_games:
        board_t, meta_t = states_to_tensor([board.fen() for board in boards])
        board_t = board_t.type(torch.float)
        meta_t = meta_t.type(torch.float)
        if t % 2 == 0:
            pred_w = model(board_t[:n_games//2], meta_t[:n_games//2])
            pred_b = opp_model(board_t[n_games//2:], meta_t[n_games//2:])
        else:
            pred_b = model(board_t[n_games//2:], meta_t[n_games//2:])
            pred_w = opp_model(board_t[:n_games//2], meta_t[:n_games//2])
        pred = torch.cat((pred_w, pred_b), dim=0)

        for n, board in enumerate(boards):
            if not board.is_game_over():
                legal_moves = list(board.legal_moves)
                valid_idxs = [move_to_action_idx(move) for move in legal_moves]
                pred_n = pred[n][valid_idxs]
                actions = torch.distributions.Categorical(logits=pred_n)
                move = legal_moves[actions.sample().item()]
                if move.promotion is not None:
                    move.promotion = 5
                if (n < n_games//2) == (t % 2 == 0):# TODO
                    moves[n].append(move.uci())
                    states[n].append(board.fen())
                board.push(move)
                if board.is_game_over():
                    n_done += 1
                    reward = reward_for_side(board, n < n_games//2)
                    rewards[n] += [reward]*len(moves[n])

                    queue.put((moves[n], states[n], rewards[n]))
        t += 1

    #for game_num in range(n_games):
    #    board = chess.Board()
    #    my_side = (epoch + game_num) % 2 == 0
    #    n_moves = 0
    #    while not board.is_game_over():
    #        if board.turn == my_side:
    #            move = choose_move(board, model, 0)
    #            # record for replay
    #            moves.append(move.uci())
    #            states.append(board.fen())
    #            n_moves += 1
    #        else:
    #            move = choose_move(board, opp_model, 0)
    #        board.push(move)

    #    reward = reward_for_side(board, my_side)
    #    rewards += [reward]*n_moves

    #queue.put((moves, states, rewards))

if __name__ == "__main__":
    use_mp = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "mp":
            print("Using multiprocessing")
            #print("warning: not working")
            use_mp = True

    mp.set_start_method("spawn")

    model = PolicyModel().to(get_device())
    model.share_memory()
    model.load_state_dict(torch.load("models/supervised.pt",
        map_location=get_device()))

    opp_model = PolicyModel().to(get_device())
    opp_model.share_memory()
    opp_model.load_state_dict(torch.load("models/supervised.pt",
        map_location=get_device()))
    opp_model_pool = []

    #opt = optim.Adam(model.parameters(), lr=1e-4)
    opt = optim.SGD(model.parameters(), lr=1e-5)
    criterion = nn.NLLLoss(reduction="none")

    for epoch in range(10000):
        print("Epoch {}".format(epoch))
        # play n games
        moves, states, rewards = [], [], []
        with torch.no_grad():
            if not use_mp:  # synchronous
                q = queue.Queue()
                run_game(game_batch_size, q, model, opp_model, epoch)
            else:           # use multiprocessing
                q = mp.Queue()
                processes = []
                for n in range(n_workers):
                    p = mp.Process(target=run_game,
                        args=(game_batch_size // n_workers,
                            q, model, opp_model, epoch))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
                #mp.spawn(run_game, args=(q, model, opp_model, epoch),
                #    nprocs=game_batch_size)

            while not q.empty():
                m, s, r = q.get()
                moves += m
                states += s
                rewards += r

        # train
        boards, metas = states_to_tensor(states)
        boards = boards.type(torch.float)
        metas = metas.type(torch.float)
        actions = actions_to_tensor(moves)
        rewards = torch.tensor(rewards, dtype=torch.float,
            device=get_device())
        loss = train(model, opt, criterion, boards, metas, actions, rewards)
        print("Loss: {:.6f}".format(loss.item()))
        print()

        torch.save(model.state_dict(), "models/reinforce.pt")

        if epoch % pool_update_dur == 0:
            opp_model_pool.append(model.state_dict())
            opp_model_pool = opp_model_pool[-max_recent_opps:]

        # pick random opponent out of pool
        params = random.choice(opp_model_pool)
        opp_model.load_state_dict(params)
