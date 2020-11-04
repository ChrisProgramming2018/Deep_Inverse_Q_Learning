from replay_buffer2 import ReplayBuffer
from iql_agent import Agent




def train(env, config):
    """

    """
    memory = ReplayBuffer((8,), (1,), config["expert_buffer_size"], config["device"])
    memory.load_memory(config["buffer_path"])
    agent = Agent(8, 1, 4, config)
    memory.idx = 10
    print("memroy idx ",memory.idx)
    if config["mode"] == "predict": 
        for t in range(config["predicter_time_steps"]):
            text = "Train Predicter {}  \ {} \r".format(t, config["predicter_time_steps"])
            print(text, end = '')
            agent.learn_predicter(memory)
            if t % 200 == 0:
                agent.test_predicter(memory)
                agent.save("pytorch_models-{trained_predicter}/")
        return 

     
    if config["mode"] == "trained_predicter":
        # load trained predicter
        print("test trained predicter")
        agent.load("pytorch_models-{trained_predicter}/")
        agent.test_predicter(memory)
        

    if config["train_predicter"]:
        for t in range(1, config["predicter_time_steps"]):
            text = "Train Predicter {}  \ {} \r".format(t, config["predicter_time_steps"])
            print(text, end = '')
            agent.learn_predicter(memory)
            if t % 2000 == 0:
                agent.test_predicter(memory)
                agent.save("pytorch_models-{trained_predicter}/")
        return
    else:
        # agent.load("pytorch_models/")
        print("continue load predicter model")
    

    
    for i_episode in range(config['episodes']):
        text = "Inverse Episode {}  \ {} \r".format(i_episode, config["episodes"])
        print(text, end = '')
        agent.learn(memory)
        if i_episode % 100 == 0:
            #agent.eval_policy(env)
            agent.test_q_value(memory)
            #agent.test_predicter(memory)
            #agent.save("pytorch_models/-{}".format(i_episode))
