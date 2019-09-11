import numpy as np 

np.random.seed(3120)

special_cards = set([1,2,3])

class Environment:
    def __init__(self):
        """
        curr_state: A 3 tuple indicating 
                    (dealer_hand, player_hand (indicating max valid sum possible), number of cards which can be converted to soft)
        """
        self.curr_state = None
        self.done = 0
        self.reset()

    def sample_card(self):
        """
        sample a card along with its colour
        """ 
        card = np.random.randint(1,11)
        if (np.random.randint(0,3)==0):
            return -card
        else:
            return card

    def sample_init_state(self):
        """
        sample initial state by picking one card each for dealer and agent
        """
        dealer_hand = self.sample_card()
        player_hand = self.sample_card()
        player_hand_softness = 0
        if (player_hand in special_cards):
            player_hand_softness=1
            player_hand+=10
        return (dealer_hand, player_hand, player_hand_softness)

    def reset(self):
        """
        reset state and done
        """
        self.curr_state = self.sample_init_state()
        self.done = 0
    
    def play_dealer(self):
        """
        play the dealer's strategy and return the reward
        """
        dealer_hand = self.curr_state[0]
        if (dealer_hand in special_cards):
            dealer_hand_softness = 1
        else:
            dealer_hand_softness = 0

        while (dealer_hand>=0 and dealer_hand<=31):
            card = self.sample_card()
            dealer_hand+=card
            if card in special_cards:
                dealer_hand += 10
                dealer_hand_softness +=1
            
            if dealer_hand>31 and dealer_hand_softness>0:
                dealer_hand_softness -= 1
                dealer_hand -= 10

            if dealer_hand>=25:
                break
        
        if dealer_hand<0 or dealer_hand>31:
            return dealer_hand,1
        
        if dealer_hand>self.curr_state[1]:
            return dealer_hand,-1
        elif dealer_hand==self.curr_state[1]:
            return dealer_hand,0
        else:
            return dealer_hand,1    
            

    def step(self, action):
        """
        action 0: stand
        action 1: hit

        Returns state, reward, done
        """

        # check if game is done
        if (self.done):
            raise ValueError("Game finished!")

        # check if agent has busted, if yes then reward -1 straightaway
        if (self.curr_state[1]<0 or (self.curr_state[1]>31 and self.curr_state[2]==0)):
            self.done=1
            return self.curr_state,-1,self.done

        # if action is to hit, sample card and see if agent busted
        if (action==1):
            card = self.sample_card()
            player_hand = self.curr_state[1] + card
            player_hand_softness = self.curr_state[2]
            if card in special_cards:
                player_hand+=10
                player_hand_softness+=1
            
            dealer_hand = self.curr_state[0]
            self.curr_state = (dealer_hand, player_hand, player_hand_softness)

            if player_hand<0:
                self.done=1
                return self.curr_state,-1,self.done
            if player_hand>31:
                if player_hand_softness>0:
                    player_hand-= 10
                    player_hand_softness-= 1
                    self.curr_state = (dealer_hand, player_hand, player_hand_softness)
                else:
                    self.done=1
                    return self.curr_state,-1,self.done
            
            return self.curr_state,0, self.done

        # if action is to stand, play as dealer and reward based on outcome
        else:
            dealer_hand, reward = self.play_dealer()
            self.curr_state = (dealer_hand, self.curr_state[1], self.curr_state[2])
            self.done=1
            return self.curr_state, reward, self.done
