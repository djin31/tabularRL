import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(3120)

special_cards = set([1,2,3])

win_state = "WIN"
lose_state = "LOSE"
draw_state = "DRAW"
win_reward = 1
lose_reward = -1
draw_reward= 0

def get_states_list():
    state_list = []
    for dealer_hand in range(1,11):
        if dealer_hand==0:
            continue
        for player_hand_softness in range(0,4):
            for hand_sum in range(-10*player_hand_softness,32):
                state_list.append((dealer_hand, hand_sum, player_hand_softness))
    state_list+=[lose_state, win_state, draw_state]
    return state_list


def get_dealer_policy():
    state_list = get_states_list()

    dealer_policy = {}
    
    for state in state_list:
        if type(state)!=tuple:
            dealer_policy[state]=0
            continue
        player_hand = state[1]  
        player_hand_softness = state[2]
        if player_hand+player_hand_softness*10 < 25:
            dealer_policy[state]=1
        else:
            dealer_policy[state]=0
    return dealer_policy

def plot_dealer_policy(q_function, policy):

    def get_q_plane(q_function, policy, softness=0):
        q_plane = np.zeros((10,32))
        for dealer_hand in range(1,11):
            for player_hand in range(-10*softness,32-10*softness):
                q_plane[dealer_hand-1][player_hand+softness*10] = q_function[((dealer_hand,player_hand, softness), policy[(dealer_hand,player_hand, softness)])]
        return q_plane
    
    fig = plt.figure(figsize=(16, 16))
    for i in range(4):
        ax = fig.add_subplot(221+i, projection='3d')
        _x = np.arange(10)
        _y = np.arange(-10*i,32-10*i)
        _xx, _yy = np.meshgrid(_x, _y)
        top = np.array(get_q_plane(q_function, policy, i))
        ax.plot_wireframe(_xx,_yy,top.T)
        ax.set_title("Special Card: "+str(i))
        ax.set_xlabel("Dealer card")
        ax.set_ylabel("Hard Sum")
        ax.set_zlabel("Value")
    plt.savefig("DealerPolicy.jpg")
    plt.show()

def plot_policy(q_function, policy, plot_type=0):

    def get_q_plane(q_function, policy, softness=0):
        q_plane = np.zeros((10,32+10*softness))
        for dealer_hand in range(1,11):
            for player_hand in range(-10*softness,32):
                q_plane[dealer_hand-1][player_hand+softness*10] = q_function[((dealer_hand,player_hand, softness), policy[(dealer_hand,player_hand, softness)])]
        return q_plane
    
    fig = plt.figure(figsize=(16, 16))
    for i in range(4):
        ax = fig.add_subplot(221+i, projection='3d')
        _x = np.arange(10)
        _y = np.arange(-10*i,32)
        _xx, _yy = np.meshgrid(_x, _y)
        top = np.array(get_q_plane(q_function, policy, i))
        if plot_type==0:
            ax.plot_wireframe(_xx,_yy,top.T)
        else:
            surf = ax.plot_surface(_xx, _yy, top.T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        ax.set_title("Special Card: "+str(i))
        ax.set_xlabel("Dealer card")
        ax.set_ylabel("Hard Sum")
        ax.set_zlabel("Value")
    fig.suptitle("State Value Function")
    plt.savefig("DealerPolicy.jpg")
    plt.show()



class Environment:
    def __init__(self,debug=False):
        """
        curr_state: A 3 tuple indicating 
                    (dealer_hand, hard_sum , number of special cards)
        """
        self.curr_state = None
        self.done = 0
        self.face_cards = set()
        self.debug = debug
        self.reset()

    def sample_card(self):
        """
        sample a card along with its colour
        """ 
        card = np.random.randint(1,11)
        if (np.random.randint(0,3)==0):
            # if self.debug:
            #     print("Drew Card",-card)
            return -card
        else:
            # if self.debug:
            #     print("Drew Card",card)
            return card

    def sample_init_state(self):
        """
        sample initial state by picking one card each for dealer and agent
        """
        dealer_hand = self.sample_card()
        player_hand = self.sample_card()
        player_hand_softness = 0
        if (player_hand in special_cards) and (player_hand not in self.face_cards):
            player_hand_softness=1
            self.face_cards.add(player_hand)

        # handling special cases
        # if player_hand<0 and dealer_hand<0:
        #     return draw_state
        # if player_hand<0:
        #     return lose_state
        # if dealer_hand<0:
        #     return win_state

        # remove redundant cases
        if (player_hand<0 or dealer_hand<0):
            return self.sample_init_state()
            
        return (dealer_hand,player_hand, player_hand_softness)

    def reset(self):
        """
        reset state and done
        """
        self.face_cards = set()
        self.curr_state = self.sample_init_state()
        self.done = 0
    
    def play_dealer(self, dealer_hand, player_hand):
        """
        play the dealer's strategy and return the reward
        """
        dealer_face_cards = set()
        if (dealer_hand in special_cards):
            dealer_hand+=10
            dealer_hand_softness = 1
            dealer_face_cards.add(dealer_hand)
        else:
            dealer_hand_softness = 0

        while (dealer_hand>=0 and dealer_hand<=31):
            card = self.sample_card()
            dealer_hand+=card
            if (card in special_cards) and (card not in dealer_face_cards):
                dealer_hand+=10
                dealer_hand_softness+=1
                dealer_face_cards.add(card)
            # if self.debug:
            #     print ("Dealer at", dealer_hand)

            if dealer_hand>=25:
                break
        
        # if the dealer can save itself then reduce special card value else go bust
        if dealer_hand>35 and dealer_hand_softness>0:
            dealer_hand-=10
        

        dealer_valid = (dealer_hand in range(0,32))
        player_valid = (player_hand in range(0,32))

        if dealer_valid==False:
            if player_valid:
                return win_reward
            else:
                return draw_reward
        if player_valid==False:
            return lose_reward

        if dealer_hand>player_hand:
            return lose_reward
        elif dealer_hand==player_hand:
            return draw_reward
        else:
            return win_reward


    def step(self, action):
        """
        action 0: stand
        action 1: hit

        Returns state, reward, done
        """

        if (self.debug):
            print("In state", self.curr_state,"with", self.face_cards)

        # check if game is done
        if (self.done):
            raise ValueError("Game finished!")

        # check if agent has reached terminal state
        if self.curr_state == lose_state:
            self.done=1
            return lose_state, lose_reward, self.done
        if self.curr_state == win_state:
            self.done=1
            return win_state, win_reward, self.done
        if self.curr_state == draw_state:
            self.done=1
            return draw_state, draw_reward, self.done

        # if action is to hit, sample card and see if agent busted
        if (action==1):
            card = self.sample_card()
            player_hand = self.curr_state[1] + card
            player_hand_softness = self.curr_state[2]
            if (card in special_cards) and (card not in self.face_cards):
                player_hand_softness+=1
                self.face_cards.add(card)
            
            dealer_hand = self.curr_state[0]
            self.curr_state = (dealer_hand, player_hand, player_hand_softness)

            if player_hand + player_hand_softness*10<0:
                self.done=1
                return lose_state,lose_reward,self.done

            if player_hand>31:
                self.done=1
                return lose_state,lose_reward,self.done
            
            return self.curr_state,0, self.done

        # if action is to stand, play as dealer and reward based on outcome
        else:
            player_hand = self.curr_state[1]
            player_hand_softness = self.curr_state[2]
            
            while(player_hand<=21 and player_hand_softness>0):
                player_hand+=10
                player_hand_softness-=1

            dealer_hand = self.curr_state[0]
            reward = self.play_dealer(dealer_hand, player_hand)
            if reward>0:
                self.curr_state = win_state
            elif reward==0:
                self.curr_state = draw_state
            else:
                self.curr_state = lose_state
            self.done=1
            return self.curr_state, reward, self.done
