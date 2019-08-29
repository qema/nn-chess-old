from common import *

for small in [True, False]:
    print("small:", small)
    cap = 100 if small else None
    states, actions = load_data(cap)
    print("saving states")
    board, meta = states_to_tensor(states, verbose=True)
    print("saving actions")
    actions_t = actions_to_tensor(actions, verbose=True)

    n = "-small" if small else ""
    torch.save(board, "proc/boards{}.pt".format(n))
    torch.save(meta, "proc/meta{}.pt".format(n))
    torch.save(actions_t, "proc/actions{}.pt".format(n))
