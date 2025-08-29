import torch

class Actor(torch.nn.Module):
    def __init__(self, state_dim, num_layers, actuator_num,action_scale=1, std_scale=1):
        super(Actor, self).__init__()
        self.num_layers = num_layers
        self.state_dim = state_dim
        self.actuator_num = actuator_num

        self.fc1_x = torch.nn.Linear(self.state_dim, self.num_layers)
        self.fc2_x = torch.nn.Linear(self.num_layers, self.num_layers)
        self.fc3_x = torch.nn.Linear(self.num_layers, self.num_layers)

        self.mean = torch.nn.Linear(self.num_layers, self.actuator_num)
        self.std = torch.nn.Parameter(torch.ones(self.actuator_num)*0.1)

        self.act_scale = action_scale
        self.std_scale = std_scale
    def forward(self, input_):
        x = torch.nn.functional.elu(self.fc1_x(input_))
        x = torch.nn.functional.elu(self.fc2_x(x))
        x = torch.nn.functional.elu(self.fc3_x(x))
        mu = torch.nn.functional.tanh(self.mean(x))

        return self.act_scale*mu, self.std_scale*self.act_scale*torch.exp(-5*self.std)
class Critic(torch.nn.Module):
    def __init__(self, state_dim, num_layers):
        super(Critic, self).__init__()
        self.num_layers = num_layers
        self.state_dim = state_dim

        self.fc1_x = torch.nn.Linear(self.state_dim, self.num_layers)
        self.fc2_x = torch.nn.Linear(self.num_layers, self.num_layers)
        self.fc3_x = torch.nn.Linear(self.num_layers, self.num_layers)
        self.fc4_x = torch.nn.Linear(self.num_layers, 1)

    def forward(self, input_):
        x = torch.nn.functional.elu(self.fc1_x(input_))
        x = torch.nn.functional.elu(self.fc2_x(x))
        x = torch.nn.functional.elu(self.fc3_x(x))
        output = self.fc4_x(x)

        return output

