from simulator import Environment, get_states_list, get_dealer_policy
import numpy as np 
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

np.random.seed(3120)

def test_policy(policy, episodes=10):
    rewards = 0
    for _ in range(episodes):
        _,reward = run_episode(policy)
        rewards += reward
    return rewards/episodes


def run_episode(policy, epsilon=0):
    episode_stack = []
    env = Environment()
    state = env.curr_state
    if (np.random.rand()<epsilon):
        episode_stack.append((state,1-policy[state]))
    else:
        episode_stack.append((state,policy[state]))
    while True:
        state, reward, done = env.step(policy[state])
        if (state!=episode_stack[-1]):
            if (np.random.rand()<epsilon):
                episode_stack.append((state,1-policy[state]))
            else:
                episode_stack.append((state,policy[state]))
        if done==1:
            return episode_stack, reward

def k_step_sarsa_online(lr=0.1, epsilon=0.1, decay_epsilon=False, number_of_episodes=100, k_step=1, plot_frequency=100):
    states = get_states_list()
    q_function = {}
    policy = {}
    scores = []

    for state in states:
        policy[state]=np.random.randint(1)
        for action in range(0,2):
            # optimistic initialization in order to encourage exploration
            q_function[(state, action)] = 0

    if decay_epsilon:
        decay_factor=0.1
    else:
        decay_factor=0

    for episode in tqdm(range(number_of_episodes)):
        env = Environment()
        step_counter=0
        done=0
        state = env.curr_state
        reward=0
        state_stack = [None]*k_step
        while(done!=True):
            if (np.random.rand()<epsilon/(1+decay_factor*episode)):
                state, reward, done = env.step(1-policy[state])
            else:
                state, reward, done = env.step(policy[state])
            update_state = state_stack[step_counter] 
            state_stack[step_counter] = state
            step_counter = (step_counter+1)%k_step
            if update_state!=None:
                q_function[(update_state, policy[update_state])]+=  lr*(q_function[(state, policy[state])] - q_function[(update_state, policy[update_state])])
                if q_function[(update_state,0)]> q_function[(update_state,1)]:
                    policy[update_state] = 0
                else:
                    policy[update_state] = 1
        
        for state in state_stack:
            if state!=None:
                q_function[(state, policy[state])]+= lr*(reward - q_function[(state, policy[state])])
                if q_function[(state,0)]> q_function[(state,1)]:
                    policy[state] = 0
                else:
                    policy[state] = 1
        if (episode%plot_frequency==0):
            scores.append(test_policy(policy,100))
    plt.plot(range(0,number_of_episodes,plot_frequency),scores)
    plt.title("Average reward vs Epsiodes")
    return policy, scores

def k_step_sarsa_offline(lr=0.1, epsilon=0.1, decay_epsilon=False, number_of_episodes=100, k_step=1, plot_frequency=100, test_episodes=100):
    states = get_states_list()
    q_function = {}
    policy = {}
    scores = []

    for state in states:
        policy[state]=np.random.randint(1)
        for action in range(0,2):
            # optimistic initialization in order to encourage exploration
            q_function[(state, action)] = 0.5

    if decay_epsilon:
        decay_factor = 0.1
    else:
        decay_factor = 0

    for episode in tqdm(range(number_of_episodes)):
        episode_stack, reward = run_episode(policy,epsilon/(1+decay_factor*episode))

        for i, (state,action) in enumerate(episode_stack[:-k_step]):
            q_function[(state,action)]+= lr*(q_function[episode_stack[i+k_step]]-q_function[(state,action)])
        for (state,action) in episode_stack[-k_step:]:
            q_function[(state,action)]+= lr*(reward-q_function[(state,action)])

        for state,action in episode_stack:
            if q_function[(state,0)]>q_function[(state,1)]:
                policy[state] = 0
            elif q_function[(state,1)]>q_function[(state,0)]:
                policy[state] = 1
        if (episode%plot_frequency==0):
            scores.append(test_policy(policy,test_episodes))
    plt.plot(range(0,number_of_episodes,plot_frequency),scores, label=str(k_step)+"_step_SARSA decay "+str(decay_epsilon))
    plt.title("Average reward vs Epsiodes")
    return policy, scores
        
def q_learning(lr=0.1, epsilon=0.1, decay_epsilon=False, number_of_episodes=100, plot_frequency=100, test_episodes=100):
    states = get_states_list()
    q_function = {}
    policy = {}
    scores = []

    for state in states:
        policy[state]=np.random.randint(1)
        for action in range(0,2):
            # optimistic initialization in order to encourage exploration
            q_function[(state, action)] = 0.5

    if decay_epsilon:
        decay_factor = 0.1
    else:
        decay_factor = 0

    for episode in tqdm(range(number_of_episodes)):

        episode_stack, reward = run_episode(policy,epsilon/(1+decay_factor*episode))

        for i, (state,action) in enumerate(episode_stack[:-1]):
            next_state = episode_stack[i+1][0]
            # change here, instead of next pair in episode stack we choose next action from policy
            q_function[(state,action)]+= lr*(q_function[(next_state,policy[next_state])]-q_function[(state,action)])
        q_function[episode_stack[-1]]+=lr*(reward-q_function[episode_stack[-1]])

        for state,action in episode_stack:
            if q_function[(state,0)]>q_function[(state,1)]:
                policy[state] = 0
            elif q_function[(state,1)]>q_function[(state,0)]:
                policy[state] = 1
        if (episode%plot_frequency==0):
            scores.append(test_policy(policy,test_episodes))
    plt.plot(range(0,number_of_episodes,plot_frequency),scores,label="q_learning")
    plt.title("Average reward vs Epsiodes")
    return policy, scores


def forward_eligibility_trace(lr=0.1, epsilon=0.1, decay_epsilon=False, number_of_episodes=100, lambd=0.5, plot_frequency=100, test_episodes=100):
    states = get_states_list()
    q_function = {}
    policy = {}
    scores = []

    for state in states:
        policy[state]=np.random.randint(1)
        for action in range(0,2):
            # optimistic initialization in order to encourage exploration
            q_function[(state, action)] = 0.5

    if decay_epsilon:
        decay_factor = 0.1
    else:
        decay_factor = 0

    for episode in tqdm(range(number_of_episodes)):
        episode_stack = []
        env = Environment()
        state = env.curr_state
        while (True):
            if (np.random.rand()<epsilon/(1+decay_factor*episode)):
                action = 1-policy[state]
            else:
                action = policy[state]
            episode_stack.append((state,action))
            
            state, reward, done = env.step(action)
            
            if (done):
                q_function[state, policy[state]]+=lr*(reward-q_function[(state, policy[state])])
                break
            
            delta = (1-lambd)*lr*(q_function[(state,action)]-q_function[episode_stack[-1]])
            
            for pair in episode_stack[::-1]:
                q_function[pair]+=delta
                delta*=lambd
        
        delta = lr*(q_function[(state,action)]-q_function[episode_stack[-1]])    
        for pair in episode_stack[::-1]:
            q_function[pair]+=delta
            delta*=lambd
    
        for state,action in episode_stack:
            if q_function[(state,0)]>q_function[(state,1)]:
                    policy[state] = 0
            elif q_function[(state,1)]>q_function[(state,0)]:
                policy[state] = 1

        if (episode%plot_frequency==0):
            scores.append(test_policy(policy,test_episodes))
            
    plt.plot(range(0,number_of_episodes,plot_frequency),scores,label="eligibility trace")
    plt.title("Average reward vs Epsiodes")
    return policy, scores

            
