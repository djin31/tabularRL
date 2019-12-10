import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from abc import abstractmethod

np.random.seed(3120)

win_state = "WIN"
lose_state = "LOSE"
draw_state = "DRAW"
win_reward = 1
lose_reward = -1
draw_reward= 0

class Environment:
    @abstractmethod
    def step(self,action):
        """
        Takes action and returns resulting state, reward, done
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset environment
        """
        pass

    @abstractmethod
    def get_action_space(self):
        """
        Return all possible actions
        """
        pass

    @abstractmethod
    def get_state_space(self):
        """
        Return all possible states
        """
        pass

class BlackJack(Environment):
    def __init__(self,debug=False):
        """
        curr_state: A 3 tuple indicating 
                    (dealer_hand, hard_sum , usable ace)
        """
        self.curr_state = None
        self.done = 0
        self.debug = debug
        self.reset()

    def sample_card(self):
        """
        sample a card 
        """ 
        card = np.random.randint(1,11)
        return card

    def sample_init_state(self):
        """
        sample initial state by picking one card each for dealer and agent
        """
        dealer_hand = self.sample_card()
        player_hand = self.sample_card()
        player_hand_softness = 0
        if player_hand==1:
            player_hand_softness = 1

        return (dealer_hand,player_hand, player_hand_softness)

    def reset(self):
        """
        reset state and done
        """
        self.curr_state = self.sample_init_state()
        self.done = 0
    
    def step(self, action):
        """
        action 0: stand
        action 1: hit

        Takes action and returns resulting state, reward, done
        """

        if (self.debug):
            print("In state", self.curr_state)

        # check if game is done
        if (self.done):
            raise ValueError("Game finished!")

        # if action is to hit, sample card and see if agent busted
        if (action==1):
            dealer_hand = self.curr_state[0]
            player_hand_softness = self.curr_state[2]

            # sample card and account for softness if it is ace
            card = self.sample_card()
            if (self.debug):
                print("Player Drew", card)
            player_hand = self.curr_state[1] + card
            if (card==1):
                player_hand_softness=1
            
            self.curr_state = (dealer_hand, player_hand, player_hand_softness)

            # return loss if player busted
            if player_hand>21:
                self.done=1
                self.curr_state=lose_state
                return lose_state,lose_reward,self.done
            
            return self.curr_state,0, self.done

        # if action is to stand, play as dealer and reward based on outcome
        else:
            player_hand = self.curr_state[1]
            player_hand_softness = self.curr_state[2]
            
            if (player_hand_softness) and (player_hand<=11):
                player_hand+=10

            dealer_hand = self.curr_state[0]
            dealer_soft = 0
            # dealer maintains only soft sum and keeps hitting until reaches soft sum of 17
            if (dealer_hand==1):
                dealer_soft = 1
                dealer_hand+=10
            while (dealer_hand<17):
                card = self.sample_card()
                if (self.debug):
                    print("Dealer Drew", card)
                dealer_hand+=card
                if (dealer_hand==1):
                    dealer_soft = 1
                    dealer_hand+=10
            # allow dealer to use softness if busted
            if (dealer_soft) and (dealer_hand>21):
                dealer_hand-=10
            
            if (player_hand>dealer_hand) or (dealer_hand>21):
                self.curr_state = win_state
                reward=win_reward
            elif player_hand==dealer_hand:
                self.curr_state = draw_state
                reward=draw_reward
            else:
                self.curr_state = lose_state
                reward = lose_reward
            self.done=1
            return self.curr_state, reward, self.done

    def get_state_space(self):
        """
        Returns all possible states
        """
        state_list = []
        for dealer_hand in range(1,11):
            for hard_sum in range(1,22):
                state_list.append((dealer_hand, hard_sum, 0))
                state_list.append((dealer_hand, hard_sum, 1))
        state_list+=[lose_state, win_state, draw_state]
        return state_list

    def get_action_space(self):
        """
        Returns all possible actions
        """
        return [0,1]

    def get_dealer_policy(self):
        """
        Returns dealer policy for blackjack
        """
        state_list = self.get_state_space()

        dealer_policy = {}
        
        for state in state_list:
            if type(state)!=tuple:
                dealer_policy[state]=0
                continue
            if state[1]+state[2]*10 < 17:
                dealer_policy[state]=1
            else:
                dealer_policy[state]=0
        return dealer_policy

    def plot_policy(self, q_function, policy):
        """
        Given q_function and policy, plots state-action value function
        """
        def get_q_plane(q_function, policy, softness=0):
            q_plane = np.zeros((10,21))
            for dealer_hand in range(1,11):
                for player_hand in range(1,21):
                    q_plane[dealer_hand-1][player_hand-1] = q_function[((dealer_hand,player_hand, softness), policy[(dealer_hand,player_hand, softness)])]
            return q_plane
        
        fig = plt.figure(figsize=(16, 16))
        for i in range(2):
            ax = fig.add_subplot(221+i, projection='3d')
            _x = np.arange(10)
            _y = np.arange(21)
            _xx, _yy = np.meshgrid(_x, _y)
            top = np.array(get_q_plane(q_function, policy, i))
            ax.plot_wireframe(_xx,_yy,top.T)
            ax.set_title("Special Card: "+str(i))
            ax.set_xlabel("Dealer card")
            ax.set_ylabel("Hard Sum")
            ax.set_zlabel("Value")
        plt.show()
