from simulator import Environment, plot_dealer_policy, get_dealer_policy, plot_policy
from evaluate_policy import mc_first_visit, mc_every_visit, k_step_td_eff
from control_policy import k_step_sarsa_online, k_step_sarsa_offline, q_learning, forward_eligibility_trace
import sys
from tqdm.auto import tqdm

def eval_dealer_policy(eval_algo=0, num_expt=1, num_episodes=1000, k_step=1):
    policy = get_dealer_policy()
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
    
    plot_dealer_policy(q_func, policy)
    return q_func

def learn_policy(algo=0, num_expt=1, num_episodes=1000, k_step=1, lr=0.1, epsilon=0.1,
                 lambd=0.5, decay_epsilon=True, plot_frequency=100, test_episodes=100,):
    if algo==0:
        policy, scores = k_step_sarsa_offline(lr=lr, epsilon=epsilon, decay_epsilon=decay_epsilon, plot_frequency=plot_frequency,
                                            number_of_episodes=num_episodes, test_episodes=test_episodes, k_step=k_step)
    elif algo==1:
        policy,scores = q_learning(lr=lr, number_of_episodes=num_episodes,decay_epsilon=decay_epsilon, plot_frequency=plot_frequency,
                                    test_episodes=test_episodes)
    else:
        policy, scores = forward_eligibility_trace(lr=lr, epsilon=epsilon, lambd=lambd, decay_epsilon=decay_epsilon, number_of_episodes=num_episodes,
                                        plot_frequency=plot_frequency, test_episodes=test_episodes)

    q = mc_every_visit(policy,num_episodes)

    plot_policy(q,policy)
    return policy,q, scores             
        