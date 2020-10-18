import random
import gym_2048
import gym
from copy import copy
import numpy as np
from multiprocessing import Pool
from itertools import product, chain
BIN_CNT = 12
BINS = [0] + [2**i for i in range(1, BIN_CNT)]
BIN_ = ['cnt' + str(i) for i in BINS]
FIELD_ = ['a' + str(i) for i in range(9)]
FIELD_NAMES = ','.join(FIELD_)
BIN_NAMES = ','.join(BIN_)
ALGO = ["random", "reward-greedy", "lookahead-1-zeros", "lookahead-2-zeros", "lookahead-3-zeros"]
MARKERS = dict(zip(ALGO, "oX^s*"))  # circle, cross, triangle, square, start
RUNS_PER_ALGO = 10


def classify_state(matrix):
    return (matrix == 0).sum()


def look_forward(env, steps=4, anchor=True):
    score = []
    board = copy(env.board)
    for dir in range(4):
        cenv = copy(env)
        next_state, reward, done, info = cenv.step(dir)
        if steps:
            # disable move if it doesn't change the field (invalid move)
            if anchor and (board == next_state).all():
                score += [-1]
            else:
                score += [sum(look_forward(cenv, steps=steps-1, anchor=False) + classify_state(next_state))]
        else:
            score += [classify_state(next_state)]
    return score


# pool.map can only pass tuples
def explore(params):
    algo, repeat_nr = params
    lookahead = ALGO.index(algo) - 1
    states = []
    env = gym.make('2048-v0', width=3, height=3)
    moves = 0
    total_reward = 0
    for step in range(9999999):
        if algo == "random":  # random is our fallback for all same anyways
            rewards = [[0]*4]
        elif algo == "reward-greedy":  # reward-greedy
            rewards = [(cenv := copy(env)).step(direction)[1] for direction in range(4)]
        else:
            rewards = look_forward(env, lookahead)
        best_move = np.random.choice(np.argwhere(rewards == np.amax(rewards)).flatten())
        next_state, reward, done, info = env.step(best_move)
        moves += 1
        total_reward += reward

        bin_cnt = [(next_state == i).sum() for i in BINS]
        zeros = classify_state(next_state)

        state = [total_reward, zeros, f"{algo}-{repeat_nr}", algo, step, total_reward, zeros]
        state += next_state.flatten().tolist() + bin_cnt
        states.append(','.join(map(str, state)))
        # env.render()
        # print({"lookahead": lookahead, "moves": moves, "reward": total_reward, "rewards": rewards, "biggest tile": next_state.max()})
        if done:
            break
    return states


if __name__ == "__main__":
    algo_repeat = list(product(ALGO, range(RUNS_PER_ALGO)))
    lines = Pool(16).map(explore, algo_repeat)
    with open(f"trace{RUNS_PER_ALGO}.csv", "w") as f:
        f.write(f"x,y,line,algo,step,reward,zeros,{FIELD_NAMES},{BIN_NAMES}\n")
        f.writelines("\n".join(chain(*lines)))
