import numpy as np 
from simulator import Environment, get_states_list

np.random.seed(3120)

def get_dealer_policy():
    state_list = get_states_list()

    dealer_policy = {}
    
    for state in state_list:
        player_hand = state[1]
        player_hand_softness = state[2]
        if player_hand+player_hand_softness*10 < 25:
            dealer_policy[state]=1
        else:
            dealer_policy[state]=0
    return dealer_policy
