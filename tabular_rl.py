from blackjack_simulator import BlackJack as Environment
from evaluate_policy import mc_first_visit, mc_every_visit, k_step_td
from control_policy import k_step_sarsa, q_learning, td_lambda
import sys
from tqdm.auto import tqdm

def eval_dealer_policy(eval_algo=0, num_expt=1, num_episodes=1000, k_step=1):
    env = Environment()
    policy = env.get_dealer_policy()
    q_func_array = []
    if eval_algo==0:
        for _ in tqdm(range(num_expt)):
            q_func_array.append(mc_first_visit(policy,num_episodes))
    elif eval_algo==1:
        for _ in tqdm(range(num_expt)):
            q_func_array.append(mc_every_visit(policy,num_episodes))
    else:
        for _ in tqdm(range(num_expt)):
            q_func_array.append(k_step_td_eff(policy=policy,number_of_episodes=num_episodes, k_step=k_step))
    
    q_func = q_func_array[0]
    for key in q_func:
        for q in q_func_array[1:]:
            q_func[key]+=q[key]
        q_func[key]/=len(q_func_array)
    
    env.plot_policy(q_func, policy)
    return q_func

def learn_policy(algo=0, num_episodes=10000, k_step=1, lr=0.1, epsilon=0.1,
                 lambd=0.5, decay_epsilon=True, plot_frequency=100, test_episodes=1000,):
    if algo==0:
        policy, scores = k_step_sarsa(lr=lr, epsilon=epsilon, decay_epsilon=decay_epsilon, plot_frequency=plot_frequency,
                                            number_of_episodes=num_episodes, test_episodes=test_episodes, k_step=k_step)
    elif algo==1:
        policy,scores = q_learning(lr=lr, number_of_episodes=num_episodes,decay_epsilon=decay_epsilon, plot_frequency=plot_frequency,
                                    test_episodes=test_episodes)
    else:
        policy, scores = td_lambda(lr=lr, epsilon=epsilon, lambd=lambd, decay_epsilon=decay_epsilon, number_of_episodes=num_episodes,
                                        plot_frequency=plot_frequency, test_episodes=test_episodes)

    q = mc_every_visit(policy,num_episodes)

    env = Environment()
    env.plot_policy(q,policy)
    return policy,q, scores             
        