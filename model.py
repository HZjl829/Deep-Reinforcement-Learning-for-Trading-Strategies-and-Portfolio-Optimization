import torch.nn as nn
import torch
from torch import linalg as LA
import numpy as np
import math
from utils import Market
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
torch.set_default_device(device)
# Set a random seed for PyTorch
seed = 42  
torch.manual_seed(seed)



        
class SubNet(nn.Module):
    def __init__(self, time, input_size=7, hidden_size=1, output_size=3, last = False):
        super().__init__()
        self.last = last
        self.output_size = output_size
        # define a layer with nn.squential
        # if time == 1, then no batch normalization 
        if time == 1:
            self.action_block = nn.Sequential(nn.Linear(input_size, hidden_size)
                                    ,nn.ReLU()
                                    ,nn.Linear(hidden_size, hidden_size)
                                    ,nn.ReLU()
                                    ,nn.Linear(hidden_size, output_size)
                                   )
        else:
            self.action_block = nn.Sequential(nn.Linear(input_size, hidden_size)
                                    ,nn.BatchNorm1d(hidden_size)
                                    ,nn.ReLU()
                                    ,nn.Linear(hidden_size, hidden_size)
                                    ,nn.BatchNorm1d(hidden_size)
                                    ,nn.ReLU()
                                    ,nn.Linear(hidden_size, output_size)
                                   )
        self.action_block.apply(self.weights_init_normal)

    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input tensor, x.shape = (B, state_dim= num_of_stocks * 2 + num_of_market_fac)
        Returns:
            torch.Tensor: The output action, output.shape = (B, 1, num_of_stocks)
        """
        if self.last:
            #enforce the equality constraint
            len = x.shape[1]
            return x[:,len - self.output_size:].unsqueeze(1)
        y = self.action_block(x)
        y = y.unsqueeze(1)
        #output shape is (batch_size, 1, num_stocks)
        # print(y.shape)
        return y
    
    def weights_init_normal(self, m):
        '''Takes in a module and initializes all linear layers with weight
           values taken from a normal distribution.'''
        classname = m.__class__.__name__
        # for every Linear layer in a model
        if classname.find('Linear') != -1:
            y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
            m.weight.data.normal_(0.0,1/np.sqrt(y))
        # m.bias.data should be 0
            m.bias.data.fill_(0)


class WealthMax(nn.Module):
    def __init__(self, market_simulator, num_stock = 2, hidden_size = 100, time_step = 3, num_brownian = 1, dt=1/252, gamma=0.1, drift_constraint = 0):

        """
        T = time_step / dt
        N = num_stock
        M = num_brownian
        """
        super().__init__()
        self.num_stock = num_stock
        self.dt = dt
        self.input_size = 1 + num_stock + num_brownian
        self.output_size = num_stock
        self.num_brownian = num_brownian
        time_step = int(time_step / dt)
        networks = [SubNet(_+1, self.input_size, hidden_size, self.output_size) for _ in range(time_step)]
        self.networks = nn.ModuleList(networks)
        self.system_param = {}
        self.gamma = gamma
        self.ce = drift_constraint   #initial drift constraint, could change over time Shape = (1,)
        self.market_simulator = market_simulator
        

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The initial state of the system, x.shape = (B, state_dim= 1 + num_stock + num_brownian)
        Returns:
            torch.Tensor: The output action for each time, output.shape = (B, T, N)
            torch.Tensor: The params for each time, is a dictionary with keys: 'mu', 'sigma', 'zeta', 'bt', 'lamda' with shape (B,N,T), (B,T,N,M), (B,T,M), (T,1), (T,1), respectively.
        """

        actions = []
        state = x
        num_brownian = self.num_brownian
        init_Bt = torch.zeros(x.shape[0], num_brownian, 1)   # the init brownian motion for zeta term
        init_param = self.system_param

        mu = init_param['geo_param'][0]
        sigma = init_param['geo_param'][1]
        zeta = init_param['zeta']
        bt = init_param['bt']
        #repeat for batch
        mu_init = mu.unsqueeze(0).expand(state.shape[0], -1, -1) #shape = (batch_size, num_stock, 1)
        sigma_init = sigma.unsqueeze(0).expand(state.shape[0], -1, -1)#shape = (batch_size, num_stock, num_brownian)
        zeta_init = zeta.unsqueeze(0).expand(state.shape[0], -1, -1) #shape = (batch_size, num_brownian, 1)
        bt_init = bt.unsqueeze(0) #shape = (1, 1)
        params = {'mu':[mu_init], 'sigma':[sigma_init.unsqueeze(1)], 'zeta':[zeta_init.transpose(1,2)], 'bt':[bt_init], 'lamda':[]}
        for network in self.networks:
            #calculate lamda 
            lamda = self.calculate_lamda(mu_init, sigma_init, zeta_init, bt_init, self.gamma, self.ce)

            # Perform a forward pass 
            action = network(state)
            # Append the action to the list of actions
            actions.append(action)

            #Update the state of the system 
            state, param, B_t = self.market_simulator(state, action, init_param, init_Bt)
            init_param = param
            init_Bt = B_t


            #update params used for the lamda calculation
            mu_init = param['geo_param'][0]      
            sigma_init = param['geo_param'][1]
            zeta_init = param['zeta']

            params['mu'].append(param['geo_param'][0])
            params['sigma'].append(param['geo_param'][1].unsqueeze(1))
            params['zeta'].append(param['zeta'].transpose(1,2))                 #shape = (batch_size, 1 num_brownian)
            params['bt'].append(param['bt'].unsqueeze(0))
            params['lamda'].append(lamda)
        # Convert the list of actions, param, into a tensor
        
        actions = torch.stack(actions, dim=1)

        # concat mu, sigam, zeta, bt for each time step
        params['mu'] = torch.stack(params['mu'][:-1], dim=2).squeeze(3)                 #shape = (batch_size, num_stock, time_step)
        params['sigma'] = torch.stack(params['sigma'][:-1], dim=1).squeeze(2)           #shape = (batch_size, time_step, num_stock, num_brownian)
        params['zeta'] = torch.stack(params['zeta'][:-1], dim=1).squeeze(2)             #shape = (batch_size, time_step, num_brownian)
        params['bt'] = torch.stack(params['bt'][:-1], dim=0).squeeze(1)                 #shape = (time_step, 1)
        params['lamda'] = torch.stack(params['lamda'], dim=0)                           #shape = (time_step, 1)
        


        # for key in params:
        #     print(key, params[key].shape)
        
        
        return actions.squeeze(2), params
        

    def set_init_system_parameter(self, **kwargs):
        """
        Args: geo_param = (mu, sigma), zeta = zeta, bt = bt
        mu.shape = (N, 1)
        sigma.shape = (N, M)
        zeta.shape = (M, 1)
        bt.shape = (1)
        """
        # input: geo_param = (miu, sigma), zeta = zeta, bt = bt

        self.system_param = kwargs
    
    

    def calculate_lamda(self, mu_init, sigma_init, zeta_init, bt_init, gamma, ce):
        """
        Args:
            mu (torch.Tensor): The mean price parameter tensor, mu.shape = (B, N, T)
            sigma (torch.Tensor): The price volatility parameter tensor, sigma.shape = (B, T, N, M)
            zeta (torch.Tensor): The random noise parameter tensor, zeta.shape = (B, T, M)
            bt (torch.Tensor): The constant change wealth tensor, bt.shape = (T, 1)
            gamma (float): The risk aversion coefficient
            ce (torch.Tensor): The drift constraint tensor, ce.shape = (1)
        Returns:
            torch.Tensor: The lagrange multiplier tensor, lamda.shape = (1)
        """
        sigma_2_inv = LA.inv(torch.bmm(sigma_init, torch.transpose(sigma_init, 1,2)))   #shape = (batch_size, num_stock, num_stock)
        zt_over_sigma = torch.bmm(sigma_2_inv, torch.bmm(sigma_init,zeta_init))         #shape = (batch_size, num_stock, 1)

        #calculate lamda 
        numerator = gamma * torch.mean((ce - bt_init + torch.bmm(torch.transpose(mu_init, 1,2), zt_over_sigma)), dim = 0)   # shape = (1, 1)
        ### IMPORTANT!!, this is currently being hard coded and not going to work for multi stock. ###
        #TODO: change this to a more general form

        # print(f'mu_init shape: {mu_init.shape}')
        # print(f'sigma_2_inv shape: {sigma_2_inv.shape}')
        denominator = torch.mean(torch.square(mu_init) * sigma_2_inv, dim = 0)       #shape = (1，1)

        lamda = numerator / denominator -1  #

        # squeeze lamda to be a scalar

        lamda = lamda.squeeze(1)

        lamda = torch.maximum(torch.zeros_like(lamda), lamda)
        return lamda
        
    def set_market_simulator(self, market_simulator):
        self.market_simulator = market_simulator

class AnaSolMaxWealth(WealthMax):
    def __init__(self, market_simulator,num_stock = 2, time_step = 3, gamma=0.1, num_brownian=1, dt=1/252):
        """
        See WealthMax for details.
        """
        super().__init__(market_simulator=market_simulator, num_stock=num_stock, num_brownian=num_brownian, time_step=time_step, dt=dt)
        self.gamma = gamma
        self.timestep = time_step

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The initial state of the system, x.shape = (B, state_dim= 1 + num_stock )
        Returns:
            torch.Tensor: The output action for each time, output.shape = (B, T, N)
            torch.Tensor: The params for each time, is a dictionary with keys: 'mu', 'sigma', 'zeta', 'bt', with shape (B,N,T), (B,T,N,M), (B,T,M), (T,1) respectively.
        """
        actions = []
        state = x
        num_brownian = self.num_brownian
        init_Bt = torch.zeros(x.shape[0], num_brownian, 1)   # the init brownian motion for zeta term
        init_param = self.system_param

        mu = init_param['geo_param'][0]
        sigma = init_param['geo_param'][1]
        zeta = init_param['zeta']
        bt = init_param['bt']
        #repeat for batch
        mu_init = mu.unsqueeze(0).expand(state.shape[0], -1, -1) #shape = (batch_size, num_stock, 1)
        sigma_init = sigma.unsqueeze(0).expand(state.shape[0], -1, -1)#shape = (batch_size, num_stock, num_brownian)
        zeta_init = zeta.unsqueeze(0).expand(state.shape[0], -1, -1) #shape = (batch_size, num_brownian, 1)
        bt_init = bt.unsqueeze(0) #shape = (1, 1)

        #init param list
        params = {'mu':[mu_init], 'sigma':[sigma_init.unsqueeze(1)], 'zeta':[zeta_init.transpose(1,2)], 'bt':[bt_init]}
        time_step = int(self.timestep / self.dt)
        for t in range(time_step):
            # Perform a forward pass 
            sigma_2_inv = LA.inv(torch.bmm(sigma_init, torch.transpose(sigma_init, 1,2)))  #shape = (batch_size, num_stock, num_stock)
            
            action = 1/(self.gamma)* torch.bmm(sigma_2_inv, mu_init) - torch.bmm(sigma_2_inv, torch.bmm(sigma_init,zeta_init)) 
            action = action.transpose(1,2)
            # Append the action to the list of actions
            actions.append(action)

            #Update the state of the system 
            state, param, B_t = self.market_simulator(state, action, init_param, init_Bt)
            init_param = param
            init_Bt = B_t

            #update params used for the optimal solution
            mu_init = param['geo_param'][0]      
            sigma_init = param['geo_param'][1]
            zeta_init = param['zeta']

            params['mu'].append(param['geo_param'][0])
            params['sigma'].append(param['geo_param'][1].unsqueeze(1))
            params['zeta'].append(param['zeta'].transpose(1,2))
            params['bt'].append(param['bt'].unsqueeze(0))

        # Convert the list of actions, param, into a tensor
        actions = torch.stack(actions, dim=1)

        # concat mu, sigam, zeta, bt for each time step
        params['mu'] = torch.stack(params['mu'][:-1], dim=2).squeeze(3)                 #shape = (batch_size, num_stock, time_step)
        params['sigma'] = torch.stack(params['sigma'][:-1], dim=1).squeeze(2)           #shape = (batch_size, time_step, num_stock, num_brownian)
        params['zeta'] = torch.stack(params['zeta'][:-1], dim=1).squeeze(2)             #shape = (batch_size, time_step, num_brownian)
        params['bt'] = torch.stack(params['bt'][:-1], dim=0).squeeze(1)                 #shape = (time_step, 1)
        params['lamda'] = torch.zeros_like(params['bt'])                                #shape = (time_step, 1)

        # for key in params:
        #     print(key, params[key].shape)

        
        return actions.squeeze(2), params
    

class AnaSolMaxWealth_driftcon(WealthMax):
    def __init__(self, market_simulator, num_stock = 1, time_step = 3, gamma=0.1, num_brownian=1, dt=1/252, drift_constraint = 0):
        """
        See WealthMax for details.
        """
        super().__init__(market_simulator= market_simulator, num_stock=num_stock, num_brownian=num_brownian, time_step=time_step, dt=dt)
        self.gamma = gamma
        self.timestep = time_step
        self.ce = drift_constraint   #initial drift constraint, could change over time Shape = (1,)
        

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The initial state of the system, x.shape = (B, state_dim= 1 + num_stock )
        Returns:
            torch.Tensor: The output action for each time, output.shape = (B, T, N)
            torch.Tensor: The params for each time, is a dictionary with keys: 'mu', 'sigma', 'zeta', 'bt', 'lamda' with shape (B,N,T), (B,T,N,M), (B,T,M), (T,1), (T,1)respectively.
        """
        actions = []
        state = x
        num_brownian = self.num_brownian
        init_Bt = torch.zeros(x.shape[0], num_brownian, 1)   # the init brownian motion for zeta term
        init_param = self.system_param

        mu = init_param['geo_param'][0]
        sigma = init_param['geo_param'][1]
        zeta = init_param['zeta']
        bt = init_param['bt']
        #repeat for batch
        mu_init = mu.unsqueeze(0).expand(state.shape[0], -1, -1) #shape = (batch_size, num_stock, 1)
        sigma_init = sigma.unsqueeze(0).expand(state.shape[0], -1, -1)#shape = (batch_size, num_stock, num_brownian)
        zeta_init = zeta.unsqueeze(0).expand(state.shape[0], -1, -1) #shape = (batch_size, num_brownian, 1)
        bt_init = bt.unsqueeze(0) #shape = (1, 1)
        
        #init param list
        params = {'mu':[mu_init], 'sigma':[sigma_init.unsqueeze(1)], 'zeta':[zeta_init.transpose(1,2)], 'bt':[bt_init], 'lamda':[]}
        time_step = int(self.timestep / self.dt)
        for t in range(time_step):
            #calculate lamda
            #sigma_inverse
            sigma_2_inv = LA.inv(torch.bmm(sigma_init, torch.transpose(sigma_init, 1,2)))   #shape = (batch_size, num_stock, num_stock)
            mu_over_sigma2_inv =  torch.bmm(sigma_2_inv, mu_init)                           #shape = (batch_size, num_stock, 1)
            zt_over_sigma = torch.bmm(sigma_2_inv, torch.bmm(sigma_init,zeta_init))         #shape = (batch_size, num_stock, 1)

            lamda = self.calculate_lamda(mu_init, sigma_init, zeta_init, bt_init, self.gamma, self.ce)
            # print(lamda)
            
            # print(lamda.shape)
            # print(lamda.type())
            action = 1/(self.gamma) * mu_over_sigma2_inv * (1+lamda) - zt_over_sigma
            # print(f'action shape: {action.shape}')

            
            action = action.transpose(1,2)
            # Append the action to the list of actions
            actions.append(action)

            #Update the state of the system 
            state, param, B_t = self.market_simulator(state, action, init_param, init_Bt)
            init_param = param
            init_Bt = B_t

            #update params used for the optimal solution
            mu_init = param['geo_param'][0]      
            sigma_init = param['geo_param'][1]
            zeta_init = param['zeta']

            params['mu'].append(param['geo_param'][0])
            params['sigma'].append(param['geo_param'][1].unsqueeze(1))
            params['zeta'].append(param['zeta'].transpose(1,2))
            params['bt'].append(param['bt'].unsqueeze(0))
            params['lamda'].append(lamda)

        # Convert the list of actions, param, into a tensor
        actions = torch.stack(actions, dim=1)

        # concat mu, sigam, zeta, bt for each time step
        params['mu'] = torch.stack(params['mu'][:-1], dim=2).squeeze(3)                 #shape = (batch_size, num_stock, time_step)
        params['sigma'] = torch.stack(params['sigma'][:-1], dim=1).squeeze(2)           #shape = (batch_size, time_step, num_stock, num_brownian)
        params['zeta'] = torch.stack(params['zeta'][:-1], dim=1).squeeze(2)             #shape = (batch_size, time_step, num_brownian)
        params['bt'] = torch.stack(params['bt'][:-1], dim=0).squeeze(1)                 #shape = (time_step, 1)
        params['lamda'] = torch.stack(params['lamda'], dim=0)                           #shape = (time_step, 1)

        # for key in params:
        #     print(key, params[key].shape)

        # print(params['lamda'])


        
        return actions.squeeze(2), params


if __name__ == '__main__':

    # model = WealthMax(num_stock=1, num_brownian=1, time_step=3, dt=1, gamma=0.1, drift_constraint=100)
    # x = torch.randn(2,3)
    # a,b,c = torch.randn(1,1), torch.randn(1,1), torch.randn(1,1)
    # d = torch.randn(1)
    # model.set_init_system_parameter(geo_param = (a,b), zeta = c, bt = d)
    
    # model(x)
    
    # model = AnaSolMaxWealth()
    # x = torch.randn(2,2)
    # model.set_init_system_parameter(geo_param = (0,1), zeta = 1)
    # model(x)

   
    market = Market(num_stock=1, num_brownian=1, batch_size=2, process_for_zeta='OU', k = 1, sig_z = 0.1)
    model = WealthMax(market,num_stock=1, num_brownian=1, time_step=3, dt=1/252, gamma=0.1, drift_constraint=100)
    x = torch.randn(2,3)
    a,b,c = torch.randn(1,1), torch.randn(1,1), torch.randn(1,1)
    d = torch.randn(1)
    model.set_init_system_parameter(geo_param = (a,b), zeta = c, bt = d)
    
    model(x)