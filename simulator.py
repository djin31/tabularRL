import numpy as np 

np.random.seed(3120)

special_cards = set([1,2,3])

win_state = (-1,31,0)
lose_state = (31,-1,0)
draw_state = (-1,-1,0)
win_reward = 1
lose_reward = -1
draw_reward= 0

def get_states_list():
    state_list = []
    for dealer_hand in range(1,11):
        if dealer_hand==0:
            continue
        for hand_sum in range(0,32):
            for player_hand_softness in range(0,4):
                state_list.append((dealer_hand, hand_sum, player_hand_softness))
    state_list+=[lose_state, win_state, draw_state]
    return state_list

class Environment:
    def __init__(self,debug=False):
        """
        curr_state: A 3 tuple indicating 
                    (dealer_hand, player_hand (indicating max valid sum possible), number of cards which can be converted to soft)
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
            if self.debug:
                print("Drew Card",-card)
            return -card
        else:
            if self.debug:
                print("Drew Card",card)
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
        if player_hand<0 and dealer_hand<0:
            return draw_state
        if player_hand<0:
            return lose_state
        if dealer_hand<0:
            return win_state
        return (dealer_hand,player_hand, player_hand_softness)

    def reset(self):
        """
        reset state and done
        """
        self.curr_state = self.sample_init_state()
        self.done = 0
        self.face_cards = set()
    
    def play_dealer(self):
        """
        play the dealer's strategy and return the reward
        """
        dealer_hand = self.curr_state[0]
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
            if self.debug:
                print ("Dealer at", dealer_hand)
            # player wins
            if dealer_hand<0:
                return dealer_hand,1

            if dealer_hand>=25:
                break
        
        # if the dealer can save itself then reduce special card value else go bust
        if dealer_hand>35 and dealer_hand_softness>0:
            dealer_hand-=10
        else:
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

        if (self.debug):
            print("In state", self.curr_state)

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

            if player_hand<0:
                if player_hand_softness>0:
                    player_hand+= 10
                    player_hand_softness-= 1
                    self.curr_state = (dealer_hand, player_hand, player_hand_softness)
                else:
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
            self.curr_state=(self.curr_state[0], player_hand, player_hand_softness)

            if player_hand==31:
                self.done=1
                return win_state, win_reward, self.done

            dealer_hand, reward = self.play_dealer()
            if reward>0:
                self.curr_state = win_state
            elif reward==0:
                self.curr_state = draw_state
            else:
                self.curr_state = lose_state
            self.done=1
            return self.curr_state, reward, self.done
