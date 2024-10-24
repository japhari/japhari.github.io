---
title: "Q-Learning Algorithm for the FrozenLake Environment"
excerpt: "In this implementation, Q-learning is applied to solve the FrozenLake-v1 environment from the Gymnasium library. FrozenLake is a grid-based environment where the agent must learn to navigate from the start to the goal while avoiding holes in the ice"
collection: teaching
# colab_url: "https://colab.research.google.com/drive/1-T78BMZQg3w9m4iSG2zF154xGAZfcuah"
github_url: "https://github.com/japhari/reinforcement-learning/blob/main/Q_Q_Learning_Algorithm_for_the_FrozenLake.ipynb"
thumbnail: "/images/publication/q-learning.png"
type: "Reinforcement Learning"
permalink: /teaching/2024-spring-teaching-1
venue: "PTIT , Department of Computer Science"
date: 2024-10-24
location: "Hanoi, Vietnam"
categories:
  - teaching
tags:
  - reinforcement learning

---
<img src="/images/animation/frozen_lake_animation.gif" alt="Frozen Lake Animation" width="640" height="360">

<!-- <video width="640" height="360" controls>
  <source src="/images/animation/frozen_lake_animation.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video> -->

## **Introduction**
**Q-learning** is a type of **Reinforcement Learning (RL)** algorithm used to teach an agent how to act optimally in an environment by learning a policy, which maps states to the best actions. The agent interacts with the environment over multiple episodes, taking actions, receiving rewards, and updating a **Q-table** that represents the value of each state-action pair. Here’s an overview of the key concepts involved:

---

### **1. Reinforcement Learning Framework**
In Q-learning, the agent operates in an environment and follows this cycle:
- **State**: The agent observes the current state of the environment.
- **Action**: The agent takes an action based on either exploration (random choice) or exploitation (choosing the best-known action).
- **Reward**: The environment provides feedback in the form of a reward based on the agent's action.
- **New State**: The environment transitions to a new state as a result of the action.

This process is repeated across **episodes**, where the agent's goal is to maximize the cumulative reward over time.

---

### **2. The Q-Table**
The **Q-table** is a matrix where each row represents a state and each column represents a possible action in that state. The values in the Q-table (Q-values) represent the expected future reward for taking a given action from a specific state.

- **Q(s, a)**: The value of taking action `a` in state `s`.
- The table is updated iteratively using the **Bellman Equation** during the learning process to reflect better estimations of future rewards.

---

### **3. Q-Learning Algorithm**
The Q-learning algorithm works as follows:
1. **Initialize** the Q-table with all values set to zero.
2. For each **episode**, repeat:
   - Start in an initial state.
   - For each **step** in the episode:
     1. Choose an action based on exploration (random action) or exploitation (action with the highest Q-value).
     2. Execute the action, move to the next state, and receive a reward.
     3. **Update the Q-value** using the formula:
    
$$
Q(s,a) = Q(s,a) + \alpha \times \left[ R + \gamma \times \max_{a'} Q(s',a') - Q(s,a) \right].
$$
       
        - `α` (learning rate): Controls how much new information overrides the old Q-value.
        - `γ` (discount factor): Reflects the importance of future rewards.
        - `R`: The immediate reward received after taking action `a` from state `s`.
        - `max(Q(s', a'))`: The maximum Q-value for the next state `s'`, considering all possible actions.

---

### **4. Exploration vs. Exploitation**
- **Exploration**: The agent takes random actions to discover new state-action pairs and learn about the environment. This is controlled by the **exploration rate** (`ε`).
- **Exploitation**: The agent uses its learned Q-table to take the best-known action. As training progresses, the agent increasingly exploits the learned policy.

The **exploration-exploitation trade-off** ensures that the agent explores initially to gather knowledge, then focuses on exploiting that knowledge to maximize rewards as it learns.

---

### **5. Updating the Q-Table**
As the agent takes actions and receives rewards, it updates the Q-table according to the **Q-learning update rule**:
- The agent adjusts the value of the current state-action pair (`Q(s, a)`) based on the immediate reward and the maximum expected future reward.
- Over many episodes, the Q-values converge to reflect the optimal policy, meaning the agent learns which action to take in each state to maximize the cumulative reward.

---

### **6. Goal of Q-Learning**
The ultimate goal of Q-learning is for the agent to develop a **policy**—a mapping from states to actions—that maximizes its total reward over time. By using the Q-table, the agent can make informed decisions and take the optimal action in any given state.

## **Implementaion**

---

### **1. Environment Setup**
The FrozenLake environment is a 4x4 grid where the agent starts at a defined position and must reach the goal while avoiding holes. Each move is an action (left, right, up, down), and the agent receives a reward of `1` for reaching the goal and `0` otherwise.

```python
env = gym.make('FrozenLake-v1', desc=None, render_mode="rgb_array", map_name="4x4", is_slippery=False)
```

- `render_mode="rgb_array"` is used to visualize the environment.
- The agent can move in 4 directions (left, right, up, down), and the environment has 16 states (one for each square on the 4x4 grid).

---

### **2. Initialize the Q-Table**
The Q-table is initialized with zeros. This table will store the **Q-values** for each state-action pair.

```python
q_table = np.zeros((state_space_size, action_space_size))
```

- The shape of the Q-table is `(16, 4)` because there are 16 states and 4 possible actions in each state.
- Initially, all values are `0` since the agent has no prior knowledge of the environment.

---

### **3. Define Hyperparameters**
Several hyperparameters control the learning process:

```python
num_episodes = 80000  # Number of episodes (learning iterations)
max_steps_per_episode = 100  # Max steps per episode

learning_rate = 0.8  # α - How much to update Q-values after each step
discount_rate = 0.99  # γ - How much to prioritize future rewards over immediate rewards

exploration_rate = 1  # Initial exploration rate (ε)
max_exploration_rate = 1  # Maximum exploration rate
min_exploration_rate = 0.001  # Minimum exploration rate
exploration_decay_rate = 0.00005  # Decay rate for exploration rate
```

- **Learning rate (α)**: Determines how much new information is used to update the Q-values.
- **Discount rate (γ)**: Balances the importance of immediate and future rewards.
- **Exploration rate (ε)**: Controls the balance between exploration (random actions) and exploitation (choosing the best-known action).

---

### **4. Q-Learning Algorithm (Training Loop)**
The main Q-learning loop runs for the specified number of episodes. In each episode, the agent:
1. **Resets the environment** to start from the initial state.
2. Takes actions based on either exploration or exploitation.
3. **Updates the Q-values** based on the rewards received and the future rewards expected from the next state.

```python
for episode in range(num_episodes):
    state = env.reset()[0]  # Reset environment to initial state
    done = False  # Keep track if the episode has ended
    rewards_current_episode = 0  # Track total rewards for the current episode
    
    for step in range(max_steps_per_episode):
        # Exploration vs. Exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])  # Exploit: choose action with highest Q-value
        else:
            action = env.action_space.sample()  # Explore: choose a random action

        # Take the action and observe the new state and reward
        new_state, reward, done, truncated, info = env.step(action)

        # Update Q-value using the Bellman Equation
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state  # Transition to the next state
        rewards_current_episode += reward  # Accumulate reward

        if done:
            break  # End episode if the agent has reached the goal or fallen into a hole

    # Decay exploration rate after each episode
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
```

- **Exploration-exploitation trade-off**: The agent selects actions either randomly (exploration) or based on the highest Q-value (exploitation). The exploration rate decays over time, meaning the agent explores more initially and exploits more as it learns.
- **Q-value update**: The Q-value for the state-action pair is updated using the **Bellman equation**. This ensures that the agent adjusts its policy based on immediate rewards and the future value of the next state.
- The process repeats for each episode, allowing the agent to learn the optimal policy over time.

---

### **5. Analyze the Results**
After training, the Q-table is analyzed to understand how well the agent learned the optimal policy. Additionally, the average rewards are calculated over chunks of episodes.

```python
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000
```

This code calculates the **average reward** the agent received every 1000 episodes, giving insights into how the agent’s performance improved over time.

---

### **6. Visualizing the Learned Policy**
Once the Q-table has been sufficiently updated, the agent can use it to navigate the environment. A few episodes are run to visualize the agent’s performance, where the agent uses the learned Q-values to take actions.

```python
for episode in range(5):  # Run 5 test episodes
    state = env.reset()[0]  # Reset environment
    done = False
    
    for step in range(max_steps_per_episode):
        clear_output(wait=True)
        screen = env.render()  # Render the environment
        plt.imshow(screen)
        plt.axis('off')  # Hide axes for better visualization
        plt.show()
        time.sleep(1)  # Slow down for visibility

        # Select action using the learned Q-table
        action = np.argmax(q_table[state,:])
        new_state, reward, done, truncated, info = env.step(action)

        if done:
            # Check if the goal was reached or if the agent fell into a hole
            if reward == 1:
                print("****You reached the goal!****")
            else:
                print("****You fell through a hole!****")
            break
        state = new_state  # Move to the next state
```

This visualization helps confirm if the agent has learned to avoid holes and reach the goal efficiently using the Q-table.

---

##  Q-Table
The **Q-table** in Q-learning is a crucial component that stores the **Q-values** for each possible state-action pair. It serves as a lookup table that guides the agent in selecting the best action to take in any given state. Over time, as the agent interacts with the environment through multiple episodes, the Q-table is updated to reflect the expected cumulative rewards for taking specific actions from each state.

### How the Q-Table Works 

Let's break down how the Q-table is used and updated based on the code.

### **1. Q-Table Initialization**

In the beginning, the Q-table is initialized to all zeros:

```python
q_table = np.zeros((state_space_size, action_space_size))
```

- `state_space_size`: The number of possible states in the environment (e.g., 16 states in a 4x4 grid for *FrozenLake*).
- `action_space_size`: The number of possible actions (e.g., 4 actions: left, right, up, down).
- The shape of the Q-table is `(state_space_size, action_space_size)`, which in this case would be `(16, 4)` for *FrozenLake*.

Initially, all Q-values are set to 0 because the agent has no knowledge of the environment and hasn't taken any actions yet.

### **2. Using the Q-Table for Action Selection (Exploitation)**

During the learning process, the agent refers to the Q-table to decide which action to take for a given state. The agent selects the action that has the highest Q-value for that state when exploiting its knowledge:

```python
action = np.argmax(q_table[state,:])
```

- `np.argmax(q_table[state,:])`: This selects the action that has the highest Q-value for the current state.
- **Example**: If the agent is in state `3`, it looks at the row corresponding to state `3` in the Q-table and selects the action (column) with the highest value.

### **3. Exploration-Exploitation Trade-off**

The agent doesn’t always exploit the Q-table. To learn effectively, it balances between **exploration** (random action) and **exploitation** (selecting the best known action from the Q-table). Exploration allows the agent to discover new state-action pairs and their rewards:

```python
exploration_rate_threshold = random.uniform(0, 1)
if exploration_rate_threshold > exploration_rate:
    action = np.argmax(q_table[state,:])  # Exploit: use the Q-table
else:
    action = env.action_space.sample()  # Explore: take a random action
```

- The agent chooses random actions (exploration) early in training to gather information about the environment. Over time, as the exploration rate decays, the agent relies more on the Q-table (exploitation).

### **4. Q-Table Update Rule**

After taking an action, the agent receives feedback from the environment in the form of a **reward** and a **new state**. The agent uses this information to update the Q-value for the state-action pair according to the **Q-learning formula**:

```python
q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
    learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
```

This equation is derived from the **Bellman equation** and consists of two parts:
- **Old Value**: `q_table[state, action] * (1 - learning_rate)`: This retains some of the old Q-value for the current state-action pair.
- **New Estimate**: `learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))`: This is the updated estimate of the Q-value based on the reward and the highest Q-value of the next state.

The update rule can be broken down as:
1. **Current Q-value**: The current estimate of the Q-value for the state-action pair.
2. **Reward**: The immediate reward received after taking the action.
3. **Future Value**: The discounted maximum future reward for the next state, `np.max(q_table[new_state, :])`. This accounts for future rewards the agent could gain by following the best possible actions from the new state.

### **5. Convergence of the Q-Table**

As the agent interacts with the environment over many episodes, the Q-table is updated repeatedly. Over time, the Q-values converge to the true values that represent the expected cumulative reward for each state-action pair. Once the Q-table has converged, the agent can use it to always select the optimal action (i.e., the action with the highest Q-value for any given state).

### **Example of Q-Table Entries**

Let's consider an example for the *FrozenLake* environment. Initially, the Q-table might look like this (assuming a 4x4 grid with 16 states and 4 actions):

| State  | Left  | Down  | Right | Up    |
|--------|-------|-------|-------|-------|
| 0      | 0     | 0     | 0     | 0     |
| 1      | 0     | 0     | 0     | 0     |
| 2      | 0     | 0     | 0     | 0     |
| 3      | 0     | 0     | 0     | 0     |
| ...    | ...   | ...   | ...   | ...   |

After a few episodes, as the agent learns, the Q-table might start to reflect higher values for actions that lead to rewards (e.g., reaching the goal) and lower or negative values for actions that lead to undesirable outcomes (e.g., falling into a hole).

| State  | Left  | Down  | Right | Up    |
|--------|-------|-------|-------|-------|
| 0      | 0.1   | 0.3   | 0.2   | 0     |
| 1      | 0     | 0.5   | 0.1   | 0.2   |
| 2      | 0.4   | 0.2   | 0.1   | 0.5   |
| 3      | 0.6   | 0.1   | 0.3   | 0.2   |
| ...    | ...   | ...   | ...   | ...   |

- **State 0**: The agent learned that moving **down** (0.3) is better than moving **right** (0.2) or **left** (0.1).
- **State 1**: Moving **down** (0.5) seems to lead to higher rewards, so the agent prefers this action.

### **6. Final Q-Table**

After the learning process (i.e., after 80,000 episodes), the Q-table will contain values that represent the optimal policy for navigating the *FrozenLake*. The agent can refer to this table to always choose the best action based on the current state.

```python
print("\n\n********Q-table********\n")
print(q_table)
```

This final Q-table reflects the optimal actions for the agent to take in each state based on the rewards and experiences accumulated over thousands of episodes.

## Hyper Parameters
 **Hyperparameters** control how the learning process takes place. These parameters affect the way the agent explores the environment, learns from experiences, and updates its Q-table. Let's go through each hyperparameter and understand its role and effect on the agent's performance.

### 1. **`num_episodes` (Number of Episodes)**
   ```python
   num_episodes = 80000
   ```
   - **Definition**: This defines how many episodes (or learning trials) the agent will run during the training phase.
   - **Effect**: A higher number of episodes gives the agent more opportunities to explore the environment and refine its Q-table. If this value is too low, the agent may not have enough time to learn an optimal policy. Too high, and it may result in excessive computation time with diminishing returns after a certain point. In your case, 80,000 episodes provide enough time for the agent to learn effectively.

### 2. **`max_steps_per_episode` (Maximum Steps per Episode)**
   ```python
   max_steps_per_episode = 100
   ```
   - **Definition**: This is the maximum number of actions the agent can take within a single episode before it is forced to stop. This parameter ensures that the episode ends even if the agent hasn't reached a terminal state (e.g., goal or fallen into a hole).
   - **Effect**: If this value is too low, the agent may not have enough time to explore the environment properly in each episode. If it’s too high, the agent might waste time on ineffective exploration, especially when it is stuck in a loop without reaching the goal. Setting it to 100 allows the agent a reasonable number of steps to either reach the goal or explore different parts of the environment.

### 3. **`learning_rate` (α - Learning Rate)**
   ```python
   learning_rate = 0.8
   ```
   - **Definition**: The learning rate controls how much of the new information overrides the old information in the Q-table. It’s the weight given to the newly observed reward when updating the Q-value.
   - **Effect**: 
     - A **higher learning rate** (closer to 1) makes the agent give more weight to recent rewards, allowing it to adapt faster to new information. However, it might also cause instability by not valuing past experiences enough.
     - A **lower learning rate** (closer to 0) makes the agent update its Q-values very slowly, which could lead to slow learning but better stability over time.
   - In your case, a learning rate of `0.8` indicates that the agent quickly incorporates new information into its Q-values, allowing it to adapt to the environment faster.

### 4. **`discount_rate` (γ - Discount Factor)**
   ```python
   discount_rate = 0.99
   ```
   - **Definition**: The discount rate controls how much importance the agent places on future rewards compared to immediate rewards. It determines the present value of future rewards.
   - **Effect**:
     - A **higher discount rate** (close to 1) means the agent values future rewards almost as much as immediate rewards. This encourages the agent to look for long-term benefits rather than just immediate gains.
     - A **lower discount rate** (closer to 0) makes the agent focus on immediate rewards, potentially leading to short-sighted decisions.
   - A discount rate of `0.99` means that the agent heavily considers future rewards when making decisions, encouraging it to choose actions that yield long-term benefits rather than short-term gains.

### 5. **`exploration_rate` (ε - Initial Exploration Rate)**
   ```python
   exploration_rate = 1
   ```
   - **Definition**: This is the probability that the agent will take a random action (explore) rather than using the Q-table to select the best-known action (exploit). The agent starts by exploring the environment.
   - **Effect**:
     - A **higher exploration rate** (closer to 1) means the agent explores more, which helps in discovering new state-action pairs and potentially better strategies.
     - A **lower exploration rate** (closer to 0) means the agent mostly exploits what it has already learned, using the Q-table to choose actions.
   - An initial exploration rate of `1` ensures that the agent fully explores the environment at the beginning of training. This helps in building a comprehensive understanding of the environment before relying on the Q-table.

### 6. **`max_exploration_rate` and `min_exploration_rate`**
   ```python
   max_exploration_rate = 1
   min_exploration_rate = 0.001
   ```
   - **Definition**: These are the upper and lower bounds for the exploration rate during training.
     - `max_exploration_rate` defines the highest probability that the agent will take a random action (initially 1, meaning the agent starts by exploring fully).
     - `min_exploration_rate` defines the lowest probability of exploration (closer to 0, where the agent relies almost entirely on the learned Q-values).
   - **Effect**: 
     - The **maximum exploration rate** ensures that the agent starts with full exploration, crucial for discovering the environment and understanding different outcomes of actions.
     - The **minimum exploration rate** ensures that as training progresses, the agent eventually stops exploring randomly and exploits the optimal policy it has learned. By setting it to `0.001`, the agent is still allowed a small amount of exploration even after many episodes.

### 7. **`exploration_decay_rate` (ε-decay rate)**
   ```python
   exploration_decay_rate = 0.00005
   ```
   - **Definition**: This controls how fast the exploration rate decays after each episode. The exploration rate decays exponentially, making the agent explore less over time and exploit more as it gains confidence in the learned Q-values.
   - **Effect**: 
     - A **higher decay rate** means that the exploration rate decreases rapidly, causing the agent to shift to exploitation (using the Q-table) earlier in training. This can lead to suboptimal learning if the agent hasn't fully explored the environment.
     - A **lower decay rate** means the agent continues exploring for a longer time before switching to exploitation, which can be useful for complex environments where extensive exploration is needed.
   - In your case, `0.00005` is a slow decay rate, meaning that the agent will explore for many episodes before focusing primarily on exploitation. This helps ensure that the agent thoroughly explores the environment and avoids getting stuck in local optima.

### **Effect of Hyperparameters on Training**

The choice of hyperparameters directly influences the agent's learning process:
- **Exploration-Exploitation Balance**: The combination of the initial exploration rate, decay rate, and minimum exploration rate determines how long the agent will explore before switching to using the Q-table. If exploration is stopped too early, the agent may miss finding the optimal policy.
- **Learning Speed and Stability**: The learning rate affects how quickly the agent updates the Q-values. A high learning rate accelerates learning but may cause instability. The discount rate ensures that the agent balances immediate rewards with long-term outcomes.
- **Training Duration**: The number of episodes and steps per episode determines how much time the agent has to learn. More episodes lead to better policies but increase training time.

In conclusion, tuning these hyperparameters is crucial for efficient learning. In your setup, the parameters are chosen to allow the agent to explore extensively early on (with a high initial exploration rate and slow decay) and learn quickly (with a relatively high learning rate) while considering future rewards (with a high discount rate).

## **Q-Learning Limitations**

1. **Large State and Action Spaces**: Q-learning struggles with environments that have large or continuous state-action spaces, as the Q-table grows exponentially, making it difficult to store and update.

2. **Slow Convergence**: Learning can be slow, especially in complex environments with sparse rewards, requiring many episodes to find an optimal policy.

3. **Lack of Generalization**: Q-learning treats each state as distinct, preventing it from generalizing knowledge across similar states, making learning inefficient in complex environments.

4. **Exploration vs. Exploitation Trade-off**: Balancing exploration (trying new actions) and exploitation (using learned knowledge) is challenging and can lead to suboptimal policies if not properly managed.

5. **Difficulty with Continuous Spaces**: Q-learning is designed for discrete spaces, making it inefficient or unusable in continuous state or action environments without modification.

6. **Hyperparameter Sensitivity**: Q-learning's performance is highly dependent on the proper tuning of hyperparameters like learning rate and exploration decay, which can be difficult to optimize.

7. **Non-Stationary Environments**: Q-learning assumes the environment is static and struggles in environments where conditions change over time.

8. **Limited to Single-Agent Scenarios**: Q-learning is not designed for multi-agent environments, where the actions of one agent affect the outcomes for others.

9. **Lack of Long-Term Planning**: The algorithm focuses on immediate state-action rewards and can fail to account for long-term strategies or delayed rewards effectively. 

Q-learning is effective in small, discrete environments but faces scalability and adaptability challenges in more complex or dynamic scenarios.


## **Conclusion**
The **Q-learning algorithm** applied to the FrozenLake environment allows the agent to learn an optimal policy by interacting with the environment, updating its Q-table, and balancing exploration and exploitation. Over thousands of episodes, the agent refines its understanding of the environment, improving its performance in reaching the goal. 