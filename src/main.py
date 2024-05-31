import numpy as np
import matplotlib.pyplot as plt

## TODO: Add additional models and integrate properly
## TODO: Add abstract methods when these models are made.  

class Admission_MDP:
  def __init__(self, goal, admission_decay, time=False, num_classes=3,
                 acceptance_probabilities=np.array([0.231]), max_time=10,
                 patience=4, new_students=np.array([3]), epsilon=0.0, alpha=0.1, gamma=0.9, q_table = {}):
    ## Fixed elements
    self.goal = goal
    self.num_classes = num_classes
    self.acceptance_probabilities = acceptance_probabilities
    self.admission_decay = admission_decay
    self.max_time = max_time
    self.patience= patience
    self.delayed_acceptance_probabilities = self._admit_decay()
    self.new_students = new_students


    ## Learning params:
    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma

    ## Initiallise Start State
    self.state = self._start_state()

    ## Initalise Q-table
    self.q_table = q_table

  def _start_state(self):
        ## State Inits elements
        pending_offers = np.zeros(self.num_classes, dtype=int) + self.new_students
        sent_offers_matrix = np.zeros((self.patience, self.num_classes), dtype=int)
        total_accepted = 0

        return (pending_offers, sent_offers_matrix, total_accepted)

  def _admit_decay(self): ## Weak intrenal method to populate the delayed_acceptance_probabilities
        delayed_acceptance_probabilities = np.zeros((self.patience, self.num_classes))
        for i in range(self.patience):
            if i == 0:
                delayed_acceptance_probabilities[i, :] = self.acceptance_probabilities
            else:
                delayed_acceptance_probabilities[i, :] = np.maximum(0, delayed_acceptance_probabilities[i-1, :] - self.admission_decay)
        
        return delayed_acceptance_probabilities



  def _reward(self, total_Accepted): ## Internal method for reward calculation
    ## L2 penalty of the difference between accepted goal and another penalty if no offers are sent.
    reward = -abs(self.goal - total_Accepted)
    return reward

  def choose_action(self):
        pending_offers, _, _ = self.state

        if np.random.rand() < self.epsilon:
            ## Exploration: choose a random action
            action = tuple(np.random.randint(0, pending + 1) for pending in pending_offers)

        else:
            ## Exploitation: choose the best action based on Q-values
            state_key = str(self.state)
            if state_key in self.q_table:
                action = tuple(max(self.q_table[state_key], key=self.q_table[state_key].get))
            else:
                action = tuple(np.random.randint(0, pending + 1, dtype=np.int64()) for pending in pending_offers)

        return action

  def transition(self, action):  ## There is an error here with
                                 ## double counting.
    ## Unpacking the state:
    pending_offers, sent_offers_matrix, total_accepted = self.state

    action = np.array(action, dtype=int)
    ## Update pending_offers
    pending_offers -= action

    pending_offers = (np.abs(pending_offers) + pending_offers) / 2

    ## Then we age all pending offers.
    sent_offers_matrix = np.roll(sent_offers_matrix, shift = 1, axis=0)

    ## Setting age current offers to 0
    sent_offers_matrix[0, :] = 0

    ## Then we add new offers to the pile
    sent_offers_matrix[0, :] = action

    ## Then we determine the transition after the aging to prevent double counting
    sent_offers_matrix = sent_offers_matrix.astype(int)
    new_admits = np.random.binomial(sent_offers_matrix, self.delayed_acceptance_probabilities)

    ## Update Total Acccepted students for reward calculation
    total_accepted += new_admits.sum()

    ## Remove the students from the pending_students:
    sent_offers_matrix -= new_admits

    ## Add new students to pending_offers
    pending_offers += self.new_students

    ## Update New Students
    new_state = (pending_offers, sent_offers_matrix, total_accepted)
    self.state = new_state
    return new_state

  def update_q_vals(self, action, reward, next_state):

    """
    Takes in the state, action, reward and the next_state do the following:
    Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a)) 
    """
    state_key = str(self.state)
    action_key = action  ## Might not work for multiples right? 
    next_state_key = str(next_state)

    ## Q Table Logic 
    if state_key not in self.q_table:  ## If the current state is not in the table, we add it
        self.q_table[state_key] = {}   ## We create a specific dictioanry for this 
    if action_key not in self.q_table[state_key]: ## We then check whether that particular action has been taken before. If it has yay, else, we add it
        self.q_table[state_key][action_key] = 0 ## We set it to 0 because we do the calculations next anyway
    if next_state_key not in self.q_table: ## 
        self.q_table[next_state_key] = {}

    best_next_action = max(self.q_table[next_state_key], key=self.q_table[next_state_key].get, default=(0,) * self.num_classes)
    best_next_q_value = self.q_table[next_state_key].get(str(best_next_action), 0)

    self.q_table[state_key][action_key] += self.alpha * (reward + self.gamma * best_next_q_value - self.q_table[state_key][action_key])  ## Double check the math but I think it is correct


  def step(self):
    ## Choosing Action
    action = self.choose_action()
    ## Transition to the next state action is chosen within transition
    new_state = self.transition(action)

    ## Reward calc
    _, _, total_accepted = new_state
    reward = self._reward(total_accepted)

    ## Update Rule
    self.update_q_vals(action, reward, new_state)

    self.max_time -= 1
    return new_state, reward, action

  def run(self, epochs = 1000):
    ## Plotting Stuff
    rewards_per_epochs = []
    ## Training stuff
    for epoch in range(epochs):

        self.state = self._start_state()
        self.max_time = 10
        total_reward = 0

        while self.max_time > 0:
          _, reward, _ = self.step()
          total_reward += reward

        rewards_per_epochs += [total_reward]
        #print(f"Epoch Number {epoch} has ended")
    plt.plot(rewards_per_epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Reward per Epoch')
    plt.show()

    ## Plotting
    return self.q_table
