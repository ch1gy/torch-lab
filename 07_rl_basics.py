import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

start = 5
end = 10
class environment():
    def __init__(self):
        self.position = start
        self.goal = end
    def reset(self):
        self.position = start
        reward = 0
        return self.position
    def step(self, action):
        if self.position != self.goal:
            if action == 0:
                self.position = self.position - 1
                self.position = max(0, min(10, self.position))
                reward = -0.01
                done = False
            elif action == 1:
                self.position = self.position + 1
                self.position = max(0, min(10, self.position))
                reward = -0.01
                done = False
        if self.position == self.goal:
            reward = 1
            done = True
        
        return self.position, reward, done

env = environment()


class Game(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 24)
        self.layer2 = nn.Linear(24, 2)
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return(x)

model = Game()
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995


for episode in range(1000):
    state = env.reset()
    done = False
    steps = 0
    while not done:
        num = random.random()
        if num < epsilon:
            action = random.randint(0, 1)
        elif num >= epsilon:
            state = torch.tensor([state], dtype=torch.float32).to(device)
            output = model(state)
            action = torch.argmax(output, dim=0).item()

        new_state, reward, done = env.step(action) 
        state_tensor = torch.tensor([state], dtype=torch.float32).to(device)
        current_q = model(state_tensor)

        target = current_q.clone().detach()
        if done:
            target[action] = reward
        else:
            new_state_tensor = torch.tensor([new_state], dtype=torch.float32).to(device)
            new_q = model(new_state_tensor).detach()
            target[action] = reward + 0.9 * torch.max(new_q).item()
        
        optimizer.zero_grad()
        loss = criterion(current_q, target)
        loss.backward()
        optimizer.step()
        steps += 1
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode}, Steps: {steps}, Epsilon: {epsilon:.3f}")
