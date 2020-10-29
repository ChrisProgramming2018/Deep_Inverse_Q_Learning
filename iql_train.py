from replay_buffer2 import ReplayBuffer
from iql_agent import Agent







def train(env, config):
    """

    """
    memory = ReplayBuffer((8,), (1,), config["expert_buffer_size"], config["device"])
    memory.load_memory(config["buffer_path"])
    agent = Agent(8, 1, 4, config)

    
    for i_episode in range(config['episodes']):
        text = "Inverse Episode {}  \ {} \r".format(i_episode, config["episodes"])
        print(text, end = '')
        agent.learn(memory)
        if i_episode % 500 == 0:
            agent.eval_policy(env)
            agent.test_q_value(memory)
