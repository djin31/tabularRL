from simulator import Environment, get_states_list, get_dealer_policy
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
    states = policy.keys()
    q_function = {}
    num_visits = {}

    for state in states:
        for action in range(0,2):
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
    states = policy.keys()
    q_function = {}

    for state in states:
        for action in range(0,2):
            q_function[(state, action)] = 0

    env = Environment(debug)

    for episode in range(number_of_episodes):
        if debug:
            print("Episode", episode)
        env.reset()

        step_counter=0
        done=0
        state = env.curr_state
        reward=0
        state_stack = [None]*k_step
        while(done!=True):
            state, reward, done = env.step(policy[state])
            update_state = state_stack[step_counter] 
            state_stack[step_counter] = state
            step_counter = (step_counter+1)%k_step
            if update_state!=None:
                q_function[(update_state, policy[update_state])]+=  lr*(q_function[(state, policy[state])] - q_function[(update_state, policy[update_state])])
        
        for state in state_stack:
            if state!=None:
                q_function[(state, policy[state])]+= lr*(reward - q_function[(state, policy[state])])
        if debug:
            for s in q_function:
                if q_function[s]!=0:
                    print(s, q_function[s])
        if debug:
            print("End Episode", episode)
    if debug:
        for s in q_function:
            if q_function[s]!=0:
                print(s, q_function[s])
    return q_function

def k_step_td_eff(policy, k_step=1, number_of_episodes=100, lr=0.1, debug=False):
    states = policy.keys()
    q_function = {}

    for state in states:
        for action in range(0,2):
            q_function[(state, action)] = 0

    for episode in range(number_of_episodes):
        if debug:
            print("Episode", episode)
        episode_stack, reward = run_episode(policy,debug)
        
        q_function[episode_stack[-1]]+=lr*(reward - q_function[episode_stack[-1]])
        T = len(episode_stack)-1
        for i, pair in enumerate(episode_stack[:-1]):
            q_function[pair] += lr*(q_function[episode_stack[min(i+k_step,T)]]-q_function[pair])
        if debug:
            print("End Episode", episode)
    if debug:
        for s in q_function:
            if q_function[s]!=0:
                print(s, q_function[s])
    return q_function
