import time
import joblib
import os
import os.path as osp
import torch
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_policy(fpath, itr='last', deterministic=False):
    # handle which epoch to load from
    if itr == 'last':
        saves = [int(x[11:-3]) for x in os.listdir(fpath) if 'saved_model' in x and len(x) > 14]
        itr = '%d' % max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d' % itr

    model = torch.load(osp.join(fpath, 'saved_model' + itr + '.pt'))

    # make function for producing an action given a single state
    if hasattr(model, 'policy'):
        get_action = lambda x: model.policy(torch.Tensor(x[None, :]).to(device))[0].detach().cpu().numpy()
    elif isinstance(model, dict):
        get_action = lambda x: model['actor'](torch.Tensor(x[None, :]).to(device))[0][0].detach().cpu().numpy()

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars' + itr + '.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy(args.fpath,
                                  args.itr if args.itr >= 0 else 'last',
                                  args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not (args.norender))
