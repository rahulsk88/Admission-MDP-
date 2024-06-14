import torch
import matplotlib.pyplot as plt

class Admission_MDP_vectorised:   ## Not sure whether this is correct
  def __init__(self, goal, admission_decay = torch.tensor([0.31, 0.12]), batch_size = 128, num_classes=2,
               acceptance_probabilities=torch.tensor([0.631, .02]), max_time=10,
               patience=4, new_students=torch.tensor([10, 7]), epsilon=0.3, alpha=0.1,
               gamma=0.9, q_table = {}):
    ## Fixed elements
    self.goal = torch.tensor([goal])
    self.num_classes = num_classes
    self.acceptance_probabilities = acceptance_probabilities
    self.admission_decay = admission_decay
    self.max_time = max_time
    self.patience= patience
    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma
    self.batch = batch_size
    self.delayed_acceptance_probabilities = self._admit_decay()
    self.new_students = new_students

    ## Initiallise Start State
    self.state = self._start_state()

    ## Initalise Q-table
    self.q_table = q_table

  def _start_state(self):
        ## State Inits elements
        pending_offers = torch.zeros((self.batch, self.num_classes), dtype=int) + self.new_students  ## (batch_size, num_classes)
        sent_offers_matrix = torch.zeros((self.batch, self.num_classes, self.patience), dtype=int)   ## (batch_size, num_classes, patience)
        total_accepted = torch.zeros(self.batch, dtype = int)                                                     ## (batch_size)

        return pending_offers, sent_offers_matrix, total_accepted  ## Can this not be an array? no because the size of each dimension is differnt

  def _admit_decay(self): ## Weak intrenal method to populate the delayed_acceptance_probabilities
        delayed_acceptance_probabilities = torch.zeros((self.num_classes, self.patience))
        for i in range(self.patience):
            if i == 0:
                delayed_acceptance_probabilities[:, i] = self.acceptance_probabilities
            else:
                delayed_acceptance_probabilities[:, i] =  delayed_acceptance_probabilities[:, i-1] - self.admission_decay
        
        delayed_acceptance_probabilities = torch.relu(delayed_acceptance_probabilities)
        delayed_acceptance_probabilities = delayed_acceptance_probabilities.expand(self.batch, self.num_classes, self.patience)

        
        return delayed_acceptance_probabilities



  def _reward(self, total_Accepted): ## Internal method for reward calculation
    ## L2 penalty of the difference between accepted goal and another penalty if no offers are sent.
    reward = -abs(self.goal - total_Accepted)  ## Broadcasting should take care of this
    return reward

  def _new_student_generator(self):
    pass


  def choose_action(self):
      pending_offers, _, _ = self.state
      random_numbers = torch.rand(self.batch)

      actions = torch.zeros((self.batch, self.num_classes), dtype=int)
      explore = random_numbers < self.epsilon

      for i in range(self.num_classes):
          actions[:, i] = torch.where(
              explore,
              torch.randint(low=0, high=pending_offers[:, i].max().item() + 1, size=(self.batch,)),
              torch.tensor([self.q_table.get(str(tuple(pending_offers[j].tolist())), {}).get((i,), 0) for j in range(self.batch)])
          )

      return actions

  def transition(self, actions):
      pending_offers, sent_offers_matrix, total_accepted = self.state
      pending_offers -= actions
      pending_offers = torch.relu(pending_offers)  # Ensure no negative values

      sent_offers_matrix = torch.roll(sent_offers_matrix, shifts=1, dims=1)
      sent_offers_matrix[:, :, 0] = actions

      generator = torch.distributions.binomial.Binomial(sent_offers_matrix, self.delayed_acceptance_probabilities)
      new_admits = generator.sample()

      total_accepted += new_admits.sum((1, 2)).int()
      sent_offers_matrix -= new_admits.int()

      pending_offers += self.new_students

      self.state = (pending_offers, sent_offers_matrix, total_accepted)
      return self.state

  def update_q_vals(self, actions, rewards, next_states):
    pending_offers, sent_offers_matrix, total_accepted = self.state
    for i in range(self.batch):
        state_key = f"{pending_offers[i].tolist()}-{sent_offers_matrix[i].view(-1).tolist()}-{total_accepted[i].item()}"
        action_key = tuple(actions[i].tolist())
        next_state_key = f"{next_states[0][i].tolist()}-{next_states[1][i].view(-1).tolist()}-{next_states[2][i].item()}"

        # Initialize dictionaries for new states and actions
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {}

        # Default to 0 if not already present
        current_q_value = self.q_table[state_key].get(action_key, 0)
        best_next_q_value = max(self.q_table[next_state_key].values(), default=0)

        # Update the Q-value using the learning rate
        updated_q_value = current_q_value + self.alpha * (rewards[i] + self.gamma * best_next_q_value - current_q_value)
        self.q_table[state_key][action_key] = updated_q_value


  def step(self):
      actions = self.choose_action()
      new_states = self.transition(actions)

      _, _, total_accepted = new_states
      rewards = self._reward(total_accepted)

      self.update_q_vals(actions, rewards, new_states)

      self.max_time -= 1
      return new_states, rewards, actions


  def run(self, epochs=1000):
      rewards_per_epoch = []
      try:
        for epoch in range(epochs):
            self.state = self._start_state()
            self.max_time = 10
            total_reward = 0

            while self.max_time > 0:
                _, rewards, _ = self.step()
                total_reward += rewards.sum().item()

            rewards_per_epoch.append(total_reward/self.batch)
      finally: 
        plt.plot(rewards_per_epoch)
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.title('Reward per Epoch')
        plt.show()

        return self.q_table

