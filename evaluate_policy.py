from simulator import Environment, get_states_list
from policy import get_dealer_policy
import numpy as np 

np.random.seed(3120)

def run_episode(policy, debug=False):
    episode_stack = []
    env = Environment(debug)
    state = env.curr_state
    episode_stack.append((state,policy[state]))
    while True:
        state, reward, done = env.step(policy[state])
        if (state!=episode_stack[-1]):
            episode_stack.append((state, policy[state]))
        if done==1:
            return episode_stack, reward


def mc_first_visit(policy, number_of_episodes=100, debug=False):
    states = policy.keys()
    q_function = {}
    num_visits = {}

    for state in states:
        for action in range(0,2):
            q_function[(state, action)] = 0
            num_visits[(state, action)] = 0
    
    for e in range(number_of_episodes):
        print ("Episode",e)
        episode_stack, reward = run_episode(policy, debug)
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


def mc_every_visit(policy, number_of_episodes=100, debug=False):
    states = policy.keys()
    q_function = {}
    num_visits = {}

    for state in states:
        for action in range(0,2):
            q_function[(state, action)] = 0
            num_visits[(state, action)] = 0
    
    for e in range(number_of_episodes):
        print ("Episode",e)
        episode_stack, reward = run_episode(policy, debug)
        print("Reward",reward)

        for pair in episode_stack:
            num_visits[pair]+=1
            n = num_visits[pair]
            q_function[pair] = (q_function[pair]*(n-1) + reward)/n
    if debug:
        for s in q_function:
            if q_function[s]!=0:
                print(s, q_function[s])

    


