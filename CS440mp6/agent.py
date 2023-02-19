import numpy as np
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO: write your function here
        
        # to train the action
        if self._train == True:
        # state = (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
            if (self.s != None) and (self.a != None):
                #update N table
                self.N[self.s][self.a] += 1

                #update Q table
                lr = self.C / (self.C + self.N[self.s][self.a])

                reward = -0.1
                if (points > self.points):
                    reward = 1
                elif dead:
                    reward = -1
                
                max_act = 0
                max_val = self.Q[s_prime][0]
                for i in [1,2,3]:
                    if self.Q[s_prime][i] > max_val:
                        max_val = self.Q[s_prime][i]
                        max_act = i

                self.Q[self.s][self.a] = self.Q[self.s][self.a] + lr * (reward + self.gamma * max_val - self.Q[self.s][self.a])

            if dead == False:
                self.points = points
                self.s = s_prime
                max_curr_act = 0
                max_curr_val = -10000
                for i in range(4):
                    if self.N[s_prime][i] < self.Ne:
                        if 1 >= max_curr_val:
                            max_curr_val = 1
                            max_curr_act = i
                    else:
                        if self.Q[s_prime][i] > max_curr_val:
                            max_curr_val = self.Q[s_prime][i]
                            max_curr_act = i
                self.a = max_curr_act
                return max_curr_act
            
            # if dead
            else:
                self.reset()
                return 0
        
        # to evaluate the action
        else:
            max_curr_act = 0
            max_curr_val = -10000
            for i in range(4):
                if self.Q[s_prime][i] > max_curr_val:
                        max_curr_val = self.Q[s_prime][i]
                        max_curr_act = i

            return max_curr_act

    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment 
        # environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y]
        # state = (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        # Find the direction of food relative to the snake head
        snake_head_x = environment[0]
        snake_head_y = environment[1]
        snake_body = environment[2]
        food_x = environment[3]
        food_y = environment[4]
        food_dir_x = 0
        if (snake_head_x > food_x):
            food_dir_x = 1
        elif (snake_head_x < food_x):
            food_dir_x = 2
        
        food_dir_y = 0
        if (snake_head_y > food_y):
            food_dir_y = 1
        elif (snake_head_y < food_y):
            food_dir_y = 2

        # Find whether there is a wall next to the snake head
        adjoining_wall_x = 0
        if (snake_head_x == 1):
            adjoining_wall_x = 1
        elif (snake_head_x == utils.DISPLAY_WIDTH - 2):
            adjoining_wall_x = 2
        
        adjoining_wall_y = 0
        if (snake_head_y == 1):
            adjoining_wall_y = 1
        elif (snake_head_y == utils.DISPLAY_HEIGHT - 2):
            adjoining_wall_y = 2

        # Checks if a grid next to the snake head contains the snake body
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0
        if (snake_head_x, snake_head_y - 1) in snake_body:
            adjoining_body_top = 1
        if (snake_head_x, snake_head_y + 1) in snake_body:
            adjoining_body_bottom = 1
        if (snake_head_x - 1, snake_head_y) in snake_body:
            adjoining_body_left = 1
        if (snake_head_x + 1, snake_head_y) in snake_body:
            adjoining_body_right = 1

        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)