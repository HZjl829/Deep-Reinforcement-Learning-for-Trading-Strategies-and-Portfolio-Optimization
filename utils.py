import torch.nn as nn
import torch
from torch import linalg as LA
import numpy as np
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
torch.set_default_device(device)


class Market(nn.Module):

    def __init__(self, num_stock, num_brownian, batch_size=128, process_for_zeta='brownian', k = 1, sig_z = 0.1, dt = 1/252):

        """
        Args:   
            num_stock (int): The number of stocks in the market
            num_brownian (int): The number of brownian motion in the market
            batch_size (int): The batch size for the simulation
            process_for_zeta (str): The process for the zeta process. Default is brownian motion, OU process is only for one dimensional case
            k (float): The parameter for the OU process. Default is 1
            sig_z (float): The volatility for the zeta OU process. Default is 0.1
            dt (float): The time step for the simulation. Default is 1/252
        """
        super(Market, self).__init__()
        self.num_stock = num_stock
        self.num_brownian = num_brownian
        self.batch_size =batch_size
        self.process = process_for_zeta
        self.sig_z = sig_z
        self.k = k
        self.dt = dt

    def zeta_simulator(self, B_t, zeta):
        """Simulate the zeta process for the market
        Args:
            B_t (torch.Tensor): The brownian motion tensor, B_t.shape = (B, M, 1)
            zeta (torch.Tensor): The zeta tensor, zeta.shape = (B, M, 1)

        Returns:
            torch.Tensor: The simulated zeta process. Shape = (B, M, 1)
        """
        if self.process == 'brownian':
            zeta = torch.bmm(torch.eye(self.num_brownian).unsqueeze(0).expand(self.batch_size, -1, -1), B_t) 
            return zeta
        elif self.process == 'OU':
           
            dB_t = torch.normal(mean = 0, std = math.sqrt(self.dt), size = (self.batch_size, self.num_brownian, 1)).to(device) #shape = (batch_size, num_brownian, 1)
            # print(((1-self.k*dt)*zeta).shape)
            # print((self.sig_z * dB_t).shape)
            return (1-self.k*self.dt)*zeta + self.sig_z * dB_t
        else:
            raise ValueError('The process for zeta is not defined')
    def B_t_simulator(self, B_t):
        """Simulate the brownian motion for the zeta process
        Args:
            B_t (torch.Tensor): The brownian motion tensor, B_t.shape = (B, M, 1)
        Returns:
            torch.Tensor: The simulated brownian motion. Shape = (B, M, 1)
        """
        dB_t = torch.normal(mean = 0, std = math.sqrt(self.dt), size = (self.batch_size, self.num_brownian, 1)).to(device) #shape = (batch_size, num_brownian, 1)
        B_t = B_t + dB_t
        return B_t


    def forward(self, state, action, param, B_t):
        """
        Args:
            state (torch.Tensor): The state tensor, state.shape = (B, state_dim= 1 + num_of stocks + num_brownian)
            action (torch.Tensor): The action tensor, action.shape = (B, 1, num_of_stocks)
            param (dict): The parameters of the system. See set_init_system_parameter for details.
            B_t (torch.Tensor): The brownian motion tensor, B_t.shape = (B, num_brownian, 1)
        Returns:
            torch.Tensor: The updated state of the system, state.shape = (B, state_dim= 1+num_of stocks+num_brownian)
            torch.Tensor: The updated parameters of the system. See set_init_system_parameter in model class for details.
            torch.Tensor: The updated brownian motion tensor, B_t.shape = (B, num_brownian, 1)
        """
  
        mu = param['geo_param'][0]      
        sigma = param['geo_param'][1]
        zeta = param['zeta']
        bt = param['bt']
        #convert bt to scalar type float
        
        #repeat each param for batch_size
        if mu.shape[0] != state.shape[0]:
            mu = mu.unsqueeze(0).expand(state.shape[0], -1, -1) #shape = (batch_size, num_stock, 1)
            sigma = sigma.unsqueeze(0).expand(state.shape[0], -1, -1) #shape = (batch_size, num_stock, num_brownian)
            zeta = zeta.unsqueeze(0).expand(state.shape[0], -1, -1) #shape = (batch_size, num_brownian, 1)
        #shape of bt is (1,1), so no need to repeat
        
        batch_size = state.shape[0]
        num_stock = self.num_stock
        num_brownian = self.num_brownian
        w_t = state[:, 0:1]
        
        s_t = state[:, 1:1+num_stock]


        
        # price change 
        dBt = torch.normal(mean=0, std=math.sqrt(self.dt), size=(batch_size, num_brownian)).unsqueeze(2).to(device) #shape = (batch_size, num_brownian, 1)
        dst = mu * self.dt + torch.bmm(sigma, dBt)    #shape = (batch_size, num_stock, 1)
        
        s_t = s_t + dst.squeeze(2)  #shape = (batch_size, num_stock)
        # wealth change
        
        dwt = torch.bmm(action,dst) + torch.bmm(torch.transpose(zeta, 1,2), dBt) + bt* self.dt # shape = (batch_size, 1, 1)
        w_t = w_t + dwt.squeeze(2) #shape = (batch_size, 1)

      
        # update mu, sigma, bt and zeta
        # to be added custom function for updating mu, sigma, bt  
        # it shouldlook similar to the zeta simulator function
        mu = mu
        sigma = sigma #shape = (batch_size, num_stock, num_brownian)
        bt = bt 

        
        zeta = self.zeta_simulator(B_t, zeta).to(device) #shape = (batch_size, num_brownian, 1)
        B_t = self.B_t_simulator(B_t).to(device) #shape = (batch_size, num_brownian, 1)
             
        zeta_for_state = zeta.squeeze(2).to(device) #shape = (batch_size, num_brownian)
    
        return torch.cat((w_t, s_t, zeta_for_state), 1), {'geo_param': (mu, sigma), 'zeta': zeta, 'bt': bt}, B_t
    


class ExecutionLoss(nn.Module):
    def __init__(self, penalty = False, target_sum = 1):
        super(ExecutionLoss, self).__init__()
        self.target_sum = target_sum
        self.penalty = penalty
        

    def forward(self, action, price):
        """
        Args:
            action (torch.Tensor): The action tensor output by the model, action.shape = (B, T, N)
            price (torch.Tensor): The price tensor, price.shape = (B, N, T)
        Returns:
            torch.Tensor: The loss mean value computed. Shape = (1,)
        """
        cost_at_times = torch.bmm(action, price) #shape (B, T, T)
     
        # print(cost_at_times)
        relavent_cost = torch.diagonal(cost_at_times, dim1=-2, dim2=-1)
        # print(relavent_cost)
        
        # print(torch.sum(relavent_cost,dim=1))
        loss = torch.sum(relavent_cost,dim=1, keepdim=True)
       
        if self.penalty:
            loss = torch.sum(relavent_cost,dim=1)+ self.penalty_equal(action, self.target_sum)
        
        # return torch.mean(loss, dim = 0)
        return torch.mean(loss, dim = 0)
    
    #TODO: Define a penalty function for the loss function both for nonnegative and equality constraints
    def penalty_non0(self, action):
        """ Penality function for nonnegative constraints"""
        # print('Penalty shape=',torch.sum(LA.vector_norm((torch.min(action, torch.zeros_like(action))), dim = 2, ord = 2), dim=1).shape )
        return torch.sum(LA.vector_norm(torch.min(action, torch.zeros_like(action)), dim = 2, ord = 2), dim=1)

    def penalty_equal(self, action, target_sum):
        """ Penality function for equality constraints"""
        sum_loss = LA.vector_norm(torch.sum(action, dim=1) - target_sum, ord = 2, dim = 1)
        return sum_loss

class Wealth(nn.Module):
    def __init__(self, dt=1/252, gamma=0.1, drift_constraint = None):
        super(Wealth, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.ct = drift_constraint          #this is the lower bound of the drift term. Shape = (T,1)

    def forward(self, action, mu, sigma,zeta, bt, lamda=None):
        """
        Args:
            action (torch.Tensor): The action tensor output by the model, action.shape = (B, T, N)
            mu (torch.Tensor): The mean price parameter tensor, mu.shape = (B, N, T)
            sigma (torch.Tensor): The price volatility parameter tensor, sigma.shape = (B, T, N, M)
            zeta (torch.Tensor): The random noise parameter tensor, zeta.shape = (B, T, M)
            bt (torch.Tensor): The constant change wealth tensor, bt.shape = (T, 1)
            lamda (torch.Tensor): The lagrange multiplier tensor, lamda.shape = (T, 1)
        Returns:
            torch.Tensor: The loss value computed. Shape = (B,1)
        """

        #repeat bt for batch size
        bt = bt.repeat(action.shape[0],1,1).squeeze(2) #shape (B, T)
        
        mu = mu.type(torch.cuda.FloatTensor)
        sigma = sigma.type(torch.cuda.FloatTensor)
        # wealth term 
        wealth = torch.bmm(action,mu)   #shape (B, T, T)
  
        relavent_wealth = torch.diagonal(wealth, dim1=-2, dim2=-1) #shape (B, T)
        #adding bt

        relavent_wealth = relavent_wealth + bt    #shape (B, T)
        wealth = torch.sum(relavent_wealth, dim=1) * self.dt   #shape (B)

        # risk term
        risk= torch.einsum('btn,btnm->btm', action, sigma)           #shape (B, T, M)
        sum = torch.sum(torch.square(LA.vector_norm(risk + zeta, dim=2, ord = 2)), dim=1) 
        risk_penalty = self.gamma/2 * self.dt * sum                     #shape (B)


        # constraints: 
        if self.ct != None:
            drift_penalty = self.drift_constraint(relavent_wealth, self.ct, lamda) # shape (B,1)

            return wealth - risk_penalty - drift_penalty

        return wealth - risk_penalty 
    
    def drift_constraint(self, relevant_wealth, ct, lamda):
        """Penalty function for drift constraint
        relevant_wealth: The drift term of wealth shape (B, T)
        ct: The lower bound of the drift term shape (T,1)
        lamda: Lagrange multiplier shape (T,1)
        
        """

        penalty = (ct - relevant_wealth)* self.dt * lamda.t() #shape (B,T)
        
        return torch.sum(penalty, dim = 1) #shape (B)


        

        

        

if __name__ == '__main__':
   
    # a = torch.randn(2,3,5)
    # b = torch.randn(2,5,3)
    
    
    # print(a)
    # print(a.shape)
    # loss = ExecutionLoss()
    # print(loss(a,b).shape)


    #test the wealth function for shape with B = 2, N=2, T=3, M=2
    wealth = Wealth(drift_constraint=0)
    
    action = torch.randn(2,3,2)
    mu = torch.randn(2,2,3)
    sigma = torch.randn(2,3,2,2)
    zeta = torch.randn(2,3,2)
    bt = torch.randn(3,1)
    lamda = torch.randn(3,1)
    print(wealth(action, mu, sigma,zeta, bt, lamda).shape)



    

   


    
    