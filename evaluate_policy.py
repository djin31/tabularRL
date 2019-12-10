from blackjack_simulator import BlackJack as Environment
import numpy as np 

np.random.seed(3120)

"""
All the algorithms work with assumption that we are working with episodic 
task and rewards indicating win and loss. 
With slight modifications, we can add in intermediate rewards as well.
"""

def run_episode(policy, debug=False):
    """
    run one complete episode using given policy
    """
    episode_stack = []
    env = Environment(debug)
    state = env.curr_state
    episode_stack.append((state,policy[state]))
    while True:
        state, reward, done = env.step(policy[state])
        episode_stack.append((state, policy[state]))
        if done==1:
            return episode_stack, reward

def mc_first_visit(policy, number_of_episodes=100, debug=False, gamma=1):
    env = Environment()
    states = env.get_state_space()
    actions = env.get_action_space()
    q_function = {}
    num_visits = {}

    for state in states:
        for action in actions:
            q_function[(state, action)] = 0
            num_visits[(state, action)] = 0
    
    for e in range(number_of_episodes):
        if debug:
            print ("Episode",e)
        episode_stack, reward = run_episode(policy, debug)
        if debug:
            print("Reward",reward)
        state_action_pairs = set(episode_stack)

        for pair in state_action_pairs:
            num_visits[pair]+=1
            n = num_visits[pair]
            q_function[pair] = (q_function[pair]*(n-1) + reward)/n
    if debug:
        for s in q_function:
            if q_function[s]!=0:
                print(s, q_function[s])
    return q_function


def mc_every_visit(policy, number_of_episodes=100, debug=False):
    env = Environment()
    states = env.get_state_space()
    actions = env.get_action_space()
    q_function = {}
    num_visits = {}

    for state in states:
        for action in actions:
            q_function[(state, action)] = 0
            num_visits[(state, action)] = 0
    
    for e in range(number_of_episodes):
        if debug:
            print ("Episode",e)
        episode_stack, reward = run_episode(policy, debug)
        if debug:
            print("Reward",reward)

        for pair in episode_stack:
            num_visits[pair]+=1
            n = num_visits[pair]
            q_function[pair] = (q_function[pair]*(n-1) + reward)/n
    if debug:
        for s in q_function:
            if q_function[s]!=0:
                print(s, q_function[s])
    return q_function

def k_step_td(policy, k_step=1, number_of_episodes=100, lr=0.1, debug=False):
    env = Environment()
    states = env.get_state_space()
    actions = env.get_action_space()
    q_function = {}

    for state in states:
        for action in actions:
            q_function[(state, action)] = 0

    for episode in range(number_of_episodes):
        if debug:
            print("Episode", episode)
        
        k_history = [] 
        env.reset()
        while True:
            state = env.curr_state
            k_history.append((state, policy[state]))
            new_state, reward, done = env.step(policy[state])
            if (len(k_history)>k_step):
                old_state_action_pair = k_history[0]
                k_history = k_history[1:]

                q_function[old_state_action_pair]+=lr*(reward+q_function[(new_state, policy[new_state])]-q_function[old_state_action_pair])
            
            if (done==1):
                break
        for state_action in k_history:
            q_function[state_action]+=lr*(reward-q_function[state_action])

    if debug:
        for s in q_function:
            if q_function[s]!=0:
                print(s, q_function[s])
    return q_function
