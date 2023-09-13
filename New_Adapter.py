class Actor_new(nn.Module):

    def __init__(self, state_dim, action_dim, action_lim):

        super(Actor_new, self).__init__()
        
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.conv1 = nn.Conv2d(state_dim[0], 16, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1,padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(131072, 16384)
        
        self.fc01 = nn.Linear(16384,4096)
        self.fc01.weight.data = fanin_init(self.fc01.weight.data.size())
    
        self.fc02 = nn.Linear(4096,512)
        self.fc02.weight.data = fanin_init(self.fc02.weight.data.size())
        
        self.fc1 = nn.Linear(512,256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(256,128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(128,64)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        self.fc4 = nn.Linear(64,action_dim)
        self.fc4.weight.data.uniform_(-EPS,EPS)

    def forward(self, state):

        x = nn.functional.relu(self.conv1(state))
        x = nn.functional.relu(self.conv2(x))
        x = self.flatten(x)
        
        x = self.fc(x)
        
        x = self.fc01(x)
        
        x = self.fc02(x)
        
        x = nn.functional.softmax(x, dim=-1)
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = F.tanh(self.fc4(x))

        #action = action * self.action_lim

        return action



