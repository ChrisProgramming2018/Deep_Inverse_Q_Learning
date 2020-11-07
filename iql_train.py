from replay_buffer import ReplayBuffer
from iql_agent import Agent
import sys




def train(env, config):
    """

    """
    memory = ReplayBuffer((8,), (1,), config["expert_buffer_size"], config["device"])
    memory.load_memory(config["buffer_path"])
    memory.idx = 1
    agent = Agent(state_size=8, action_size=4,  config=config) 
    
    #for i in range(10):
    #    print("state", memory.obses[i])
    # sys.exit()
    print("memroy idx ",memory.idx)
    if config["mode"] == "predict":
        for t in range(config["predicter_time_steps"]):
            text = "Train Predicter {}  \ {} \r".format(t, config["predicter_time_steps"])
            print(text, end = '')
            agent.learn_predicter(memory)
            if t % 500 == 0:
                agent.test_predicter(memory)
                agent.save("pytorch_models-{trained_predicter}/")
        return

    
    if config["mode"] == "iql":
        agent.load("pytorch_models-{trained_predicter}/")
        agent.test_predicter(memory)
        for t in range(config["predicter_time_steps"]):
            text = "Train IQL {}  \ {} \r".format(t, config["predicter_time_steps"])
            print(text, end = '')
            agent.learn(memory)
            if t % 100 == 0:
                agent.test_q_value(memory)


    if config["mode"] == "dqn":
        print("mode dqn")
        agent.dqn_train()
        return
