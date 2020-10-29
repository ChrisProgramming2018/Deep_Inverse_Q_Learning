import sys
import json
import gym
import argparse
from utils import mkdir
from iql_train import train

def main(args):
    """ """


    with open (args.param, "r") as f:
        param = json.load(f)
    print("use the env {} ".format(param["env_name"]))
    print(param)
    env = gym.make(param["env_name"])

    train(env, param)










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--locexp', default="test", type=str)
    parser.add_argument('--lr_iql_q', default=1e-3, type=float)
    parser.add_argument('--lr_iql_r', default=1e-3, type=float)
    parser.add_argument('--lr_q_sh', default=1e-3, type=float)
    parser.add_argument('--freq_q', default=1, type=int)
    parser.add_argument('--mode', default="train q table", type=str)
    arg = parser.parse_args()
    mkdir("", arg.locexp)
    main(arg)
