import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas 
import numpy as np
import matplotlib.pyplot as plt

all_t = pandas.read_csv('index_ready_INA.csv')['t']
dt = 1
Dt = torch.tensor(dt)
index_start = 365 #1 Mar 2021
index_end = 549 #1 Sep 2021
additional_days = 30
N_p = 274*(10**6) #Initial population at t=0 (1 Mar 2020), using estimates of 1 January 2020 population (WPP UN data)
t_additional = np.array(list(all_t)[index_start:index_end+1+additional_days])
t_data = np.array(list(all_t)[index_start:index_end+1])
n = len(t_data)
n_add = len(t_additional)
torch.manual_seed(1)

Total_case = np.array(list(pandas.read_csv('Total_Case_ready_INA.csv')['Total Case'][index_start:index_end+1]))
Total_case_add = np.array(list(pandas.read_csv('Total_Case_ready_INA.csv')['Total Case'][index_start:index_end+1+additional_days]))

#Setup and Normalize Epidemic Data
S_data = np.array(list(pandas.read_csv('S_ready_INA.csv')['S'][index_start:index_end+1]))/N_p
I_data = np.array(list(pandas.read_csv('I_ready_INA.csv')['I'][index_start:index_end+1]))/N_p
R_data = np.array(list(pandas.read_csv('R_ready_INA.csv')['R'][index_start:index_end+1]))/N_p
D_data = np.array(list(pandas.read_csv('D_ready_INA.csv')['D'][index_start:index_end+1]))/N_p
S_data_add = np.array(list(pandas.read_csv('S_ready_INA.csv')['S'][index_start:index_end+1+additional_days]))/N_p
I_data_add = np.array(list(pandas.read_csv('I_ready_INA.csv')['I'][index_start:index_end+1+additional_days]))/N_p
R_data_add = np.array(list(pandas.read_csv('R_ready_INA.csv')['R'][index_start:index_end+1+additional_days]))/N_p
D_data_add = np.array(list(pandas.read_csv('D_ready_INA.csv')['D'][index_start:index_end+1+additional_days]))/N_p

mean_S = np.mean(S_data); std_S = np.std(S_data)
mean_I = np.mean(I_data); std_I = np.std(I_data)
mean_R = np.mean(R_data); std_R = np.std(R_data)
mean_D = np.mean(D_data); std_D = np.std(D_data)

S_tensor = (S_data-mean_S)/std_S
I_tensor = (I_data-mean_I)/std_I
R_tensor = (R_data-mean_R)/std_R
D_tensor = (D_data-mean_D)/std_D
### Ends here


### Setup Data to Feed the Input Layer of NN
n_data = len(S_data)
input_tensor = torch.tensor(np.array([[S_tensor[i], I_tensor[i], R_tensor[i], D_tensor[i]] for i in range(0, n_data-1)], dtype=np.float32))
n_input_tensor = len(input_tensor)
### Ends here


S_data = torch.tensor(S_data, dtype=torch.float32).reshape(-1, 1)
I_data = torch.tensor(I_data, dtype=torch.float32).reshape(-1, 1)
R_data = torch.tensor(R_data, dtype=torch.float32).reshape(-1, 1)
D_data = torch.tensor(D_data, dtype=torch.float32).reshape(-1, 1)


### ParameterModel
class ParameterModel(nn.Module):
    def __init__(self, F=30):
        super(ParameterModel, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.n_data = n_input_tensor
        self.F = F
        self.n_coef_fourier = 2*self.F
        self.n_input_dim = 4
        self.layers_size = [self.n_input_dim, 9, 19, 1]

        self.W1_rows = [nn.Parameter(25*torch.randn([1, self.layers_size[0]*self.F])) for i in range(self.layers_size[1])]
        self.W1_matrix = [row.expand((self.n_data, self.layers_size[0]*self.F)) for row in self.W1_rows]
        self.W1 = torch.stack(self.W1_matrix, dim=0)

        self.W2_rows = [nn.Parameter(25*torch.randn([1, self.layers_size[1]*self.F])) for i in range(self.layers_size[2])]
        self.W2_matrix = [row.expand((self.n_data, self.layers_size[1]*self.F)) for row in self.W2_rows]
        self.W2 = torch.stack(self.W2_matrix, dim=0)

        self.W3_rows = [nn.Parameter(25*torch.randn([1, self.layers_size[2]*self.F])) for i in range(self.layers_size[3])]
        self.W3_matrix = [row.expand((self.n_data, self.layers_size[2]*self.F)) for row in self.W3_rows]
        self.W3 = torch.stack(self.W3_matrix, dim=0)

        self.Coeff1_cos = nn.Parameter(0.01*torch.randn([self.layers_size[1], self.F*self.layers_size[0], 1]))
        self.Coeff1_sin = nn.Parameter(0.01*torch.randn([self.layers_size[1], self.F*self.layers_size[0], 1]))
        self.Coeff2_cos = nn.Parameter(0.01*torch.randn([self.layers_size[2], self.F*self.layers_size[1], 1]))
        self.Coeff2_sin = nn.Parameter(0.01*torch.randn([self.layers_size[2], self.F*self.layers_size[1], 1]))
        self.Coeff3_cos = nn.Parameter(0.01*torch.randn([self.layers_size[3], self.F*self.layers_size[2], 1]))
        self.Coeff3_sin = nn.Parameter(0.01*torch.randn([self.layers_size[3], self.F*self.layers_size[2], 1]))
        
    def cosine_base(self, W, x):
        return torch.cos(W*x)

    def sine_base(self, W, x):
        return torch.sin(W*x)

    def forward_layer(self, x, coeff_cos, coeff_sin, freq):
        xx = torch.repeat_interleave(x, repeats = self.F, dim=1)
        x = torch.matmul(self.cosine_base(freq, xx) , coeff_cos) + torch.matmul(self.sine_base(freq, xx) , coeff_sin)
        return x.reshape((x.shape[0], x.shape[1])).T

    def forward(self, x):
        x = self.forward_layer(x, self.Coeff1_cos, self.Coeff1_sin, self.W1)
        x = self.forward_layer(x, self.Coeff2_cos, self.Coeff2_sin, self.W2)
        x = self.forward_layer(x, self.Coeff3_cos, self.Coeff3_sin, self.W3)
        return self.sigmoid(x)

    def forward_new(self, x, n_data):
        W1_matrix2 = [row.expand((n_data, self.layers_size[0]*self.F)) for row in self.W1_rows]
        W1_2 = torch.stack(W1_matrix2, dim=0)

        W2_matrix2 = [row.expand((n_data, self.layers_size[1]*self.F)) for row in self.W2_rows]
        W2_2 = torch.stack(W2_matrix2, dim=0)

        W3_matrix2 = [row.expand((n_data, self.layers_size[2]*self.F)) for row in self.W3_rows]
        W3_2 = torch.stack(W3_matrix2, dim=0)

        x = self.forward_layer(x, self.Coeff1_cos, self.Coeff1_sin, W1_2)
        x = self.forward_layer(x, self.Coeff2_cos, self.Coeff2_sin, W2_2)
        x = self.forward_layer(x, self.Coeff3_cos, self.Coeff3_sin, W3_2)
        return self.sigmoid(x)        


## Instantiate the seven surrogate neural networks
beta_model = ParameterModel(F=30)
gamma_model = ParameterModel(F=30)
mu_model = ParameterModel(F=30)
## Ends here


## Runge-Kutta (RK4) Implementation Functions
def f_S(S,I,R,D,beta,gamma,mu):
    return (-beta*S*(I))

def f_I(S,I,R,D,beta,gamma,mu):
    return  (beta*S*(I) - gamma*I -mu*I )

def f_R(S,I,R,D,beta,gamma,mu):
    return (gamma*I)

def f_D(S,I,R,D,beta,gamma,mu):
    return (mu*I)

def F(S,I,R,D,beta,gamma,mu,dt):
    return dt*f_S(S,I,R,D,beta,gamma,mu),\
           dt*f_I(S,I,R,D,beta,gamma,mu),\
           dt*f_R(S,I,R,D,beta,gamma,mu),\
           dt*f_D(S,I,R,D,beta,gamma,mu)
    
def RK4(S,I,R,D,beta,gamma,mu,dt):
    K1 = F(S,I,R,D,beta,gamma,mu,dt)
    K2 = F(S + K1[0]/2, I + K1[1]/2, R + K1[2]/2, D + K1[3]/2, beta, gamma, mu, dt)
    K3 = F(S + K2[0]/2, I + K2[1]/2, R + K2[2]/2, D + K2[3]/2, beta, gamma, mu, dt)
    K4 = F(S + K3[0], I + K3[1], R + K3[2], D + K3[3], beta, gamma, mu, dt)
    return S + (1/6)*(K1[0] + K4[0]) + (1/3)*(K2[0]+K3[0]), \
           I + (1/6)*(K1[1] + K4[1]) + (1/3)*(K2[1]+K3[1]), \
           R + (1/6)*(K1[2] + K4[2]) + (1/3)*(K2[2]+K3[2]), \
           D + (1/6)*(K1[3] + K4[3]) + (1/3)*(K2[3]+K3[3])
    
      
# ------------ JOINT LOSS FUNCTIONS --------------------------
omega_S = 1
omega_I = 500
omega_R = 500
omega_D = 20000
def loss_function(S_data, I_data, R_data, D_data, \
                  beta_model_all, gamma_model_all, mu_model_all):
    RK4_result = RK4(S_data[0:-1], I_data[0:-1], R_data[0:-1], D_data[0:-1], \
                     beta_model_all, gamma_model_all, mu_model_all, Dt)
    loss_S = (omega_S)*( torch.sqrt(torch.mean((S_data[1:] - RK4_result[0])**2)))    
    loss_I = (omega_I)*( torch.sqrt(torch.mean((I_data[1:] - RK4_result[1])**2)))
    loss_R = (omega_R)*( torch.sqrt(torch.mean((R_data[1:] - RK4_result[2])**2))) 
    loss_D = (omega_D)*( torch.sqrt(torch.mean((D_data[1:] - RK4_result[3])**2)))
    total_loss = (loss_S + loss_I + loss_R + loss_D)
    return total_loss, loss_S, loss_I, loss_R, loss_D
# --------------- ENDS HERE ---------------------------------------------


## ------------------ FUNCTION FOR TRAINING THE NEURAL NETWORKS ---------------------------------------------------
def train_SIRD_model(max_epoch, beta_model, gamma_model, mu_model):

    optimizer = optim.Adam(list(beta_model.parameters()) + list(gamma_model.parameters()) + list(mu_model.parameters()), lr=0.00002)
    total_loss_values = []
    log_loss_values = []
    log_loss_values_S = []; log_loss_values_I = []; log_loss_values_R = []; log_loss_values_D = []
    
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        beta_model_all = beta_model(input_tensor)
        gamma_model_all = gamma_model(input_tensor)
        mu_model_all = mu_model(input_tensor)
        
        loss, loss_S, loss_I, loss_R, loss_D = loss_function(S_data, I_data, R_data, D_data, \
                                                             beta_model_all, gamma_model_all, mu_model_all)
        true_loss = loss.item()
        loss.backward()    
        optimizer.step()
        loss, loss_S, loss_I, loss_R, loss_D = loss_function(S_data, I_data, R_data, D_data, \
                                                             beta_model_all, gamma_model_all, mu_model_all)

        total_loss_values.append(loss.item())
        log_loss_values.append(np.log(loss.item()))
        log_loss_values_S.append(torch.log(loss_S).item()); log_loss_values_I.append(torch.log(loss_I).item())
        log_loss_values_R.append(torch.log(loss_R).item()); log_loss_values_D.append(torch.log(loss_D).item())

        if epoch % 20 == 0:
            print(f"{epoch}, S:{loss_S.item()}, I:{loss_I.item()}, R:{loss_R.item()}, D:{loss_D.item()}, all:{loss.item()}")
    print(f"S:{loss_S.item()},I:{loss_I.item()},R:{loss_R.item()},D:{loss_D.item()}, all:{loss.item()}")
    
    return total_loss_values, log_loss_values, log_loss_values_S, log_loss_values_I, log_loss_values_R, log_loss_values_D
#####--------------------------------------- FUNCTION END HERE ---------------------------------------


##---------------- TRAIN THE NEURAL NETWORK -------------------------
n_epoch = 4000

import time
start_tm = time.time()
loss_list, log_loss_list, log_loss_list_S, log_loss_list_I, log_loss_list_R, log_loss_list_D = train_SIRD_model(n_epoch, beta_model, gamma_model, mu_model)
end_tm = time.time()
print("elapsed time = " + str(end_tm-start_tm))
#----------------- TRAIN ENDS HERE -----------------


### MODEL FITTING BEFORE PREDICTION

beta_model.eval(); gamma_model.eval(); mu_model.eval()
beta = beta_model(input_tensor).detach().numpy()
gamma = gamma_model(input_tensor).detach().numpy()
mu = mu_model(input_tensor).detach().numpy()
S = [S_data[0].detach().numpy()]; I = [I_data[0].detach().numpy()]; R = [R_data[0].detach().numpy()]; D = [D_data[0].detach().numpy()]
for i in range(n_input_tensor):
    K1 = F(S[i],I[i],R[i],D[i], beta[i], gamma[i], mu[i], dt)
    K2 = F(S[i]+K1[0]/2, I[i]+K1[1]/2, R[i]+K1[2]/2, D[i]+K1[3]/2, beta[i], gamma[i], mu[i], dt)
    K3 = F(S[i]+K2[0]/2, I[i]+K2[1]/2, R[i]+K2[2]/2, D[i]+K2[3]/2, beta[i], gamma[i], mu[i], dt)
    K4 = F(S[i]+K3[0], I[i]+K3[1], R[i]+K3[2], D[i]+K3[3], beta[i], gamma[i], mu[i], dt)
    S.append( S[i] + (1/6)*(K1[0] + K4[0]) + (1/3)*(K2[0]+K3[0]) )
    I.append( I[i] + (1/6)*(K1[1] + K4[1]) + (1/3)*(K2[1]+K3[1]) )
    R.append( R[i] + (1/6)*(K1[2] + K4[2]) + (1/3)*(K2[2]+K3[2]) )
    D.append( D[i] + (1/6)*(K1[3] + K4[3]) + (1/3)*(K2[3]+K3[3]) )
    
fig,ax = plt.subplots(2,2);

ax[0][0].plot(t_data, N_p*S_data.detach().numpy(), 'o', color='orange', lw=1, ms=3)
ax[0][0].plot(t_data, np.array(S)*N_p,'-',color='blue',lw=1); ax[0][0].set_xlim(t_data[0], t_data[-1]); ax[0][0].set_xlabel(r"$t$"); ax[0][0].set_title(r"$S(t)$ v.s. $S_{model}(t)$", fontsize=12)

ax[0][1].plot(t_data, N_p*I_data.detach().numpy(), 'o', color='orange', lw=1, ms=3)
ax[0][1].plot(t_data, np.array(I)*N_p,'-',color='blue',lw=1); ax[0][1].set_xlim(t_data[0], t_data[-1]); ax[0][1].set_xlabel(r"$t$"); ax[0][1].set_title(r"$I(t)$ v.s. $I_{model}(t)$", fontsize=12)

ax[1][0].plot(t_data, N_p*D_data.detach().numpy(), 'o', color='orange', lw=1, ms=3)
ax[1][0].plot(t_data, np.array(D)*N_p,'-',color='blue',lw=1)
ax[1][0].set_xlim(t_data[0], t_data[-1]); ax[1][0].set_xlabel(r"$t$"); ax[1][0].set_title(r"$D(t)$ v.s. $D_{model}(t)$", fontsize=12)

ax[1][1].plot(t_data, N_p*R_data.detach().numpy(), 'o', color='orange', lw=1, ms=3)
ax[1][1].plot(t_data, np.array(R)*N_p,'-',color='blue',lw=1)
ax[1][1].set_xlim(t_data[0], t_data[-1]); ax[1][1].set_xlabel(r"$t$"); ax[1][1].set_title(r"$R(t)$ v.s. $R_{model}(t)$", fontsize=12)

fig.suptitle('Model Fitting')
fig.tight_layout(pad=0.5); fig.show()
### END OF MODEL FITTING BEFORE PREDICTION


##PREDICTION OF CUMULATIVE NUMBER OF CASES C(t)
Total_case_pred = [Total_case_add[n_data-1]]

x_in = torch.tensor([ [ (S_data_add[n_data-1]-mean_S)/std_S, \
                        (I_data_add[n_data-1]-mean_I)/std_I, \
                        (R_data_add[n_data-1]-mean_R)/std_R, \
                        (D_data_add[n_data-1]-mean_D)/std_D] ], dtype = torch.float).reshape((1,4))
beta = [ 0.17*beta_model.forward_new( x_in, n_data = 1 ).detach().numpy()[0][0] ]
for i in range(additional_days):
    S = S_data_add[n_data-1+i]
    I = I_data_add[n_data-1+i]
    K1 = F(S, I, 0, 0, beta[-1], 0, 0, dt)
    K2 = F(S+K1[0]/2, I+K1[1]/2, 0, 0, beta[-1], 0, 0, dt)
    K3 = F(S+K2[0]/2, I+K2[1]/2, 0, 0, beta[-1], 0, 0, dt)
    K4 = F(S+K3[0], I+K3[1], 0, 0, beta[-1], 0, 0, dt)
    Total_case_pred.append( Total_case_add[n_data-1+i] + N_p*(((1/6)*(K1[1] + K4[1]) + (1/3)*(K2[1]+K3[1])) + ((1/6)*(K1[2] + K4[2]) + (1/3)*(K2[2]+K3[2])) + \
                                                            ((1/6)*(K1[3] + K4[3]) + (1/3)*(K2[3]+K3[3]))) )

    x_in = torch.tensor([ [ (S_data_add[n_data+i]-mean_S)/std_S, \
                            (I_data_add[n_data+i]-mean_I)/std_I, \
                            (R_data_add[n_data+i]-mean_R)/std_R, \
                            (D_data_add[n_data+i]-mean_D)/std_D] ], dtype = torch.float).reshape((1,4))
    beta.append( 0.17*beta_model.forward_new( x_in, n_data = 1 ).detach().numpy()[0][0]  )

err_C = np.sqrt(sum([(Total_case_pred[i]-Total_case_add[n_data-1+i])**2 for i in range(1, additional_days)])/(additional_days-1))
print("Prediction error = {}".format(err_C))

fig, ax = plt.subplots()
ax.plot(t_additional, Total_case_add, 'o', color='orange', lw=1, ms=3);
ax.plot(t_additional[-additional_days-1:], Total_case_pred,'-o',color='blue',lw=1);
ax.set_xlim(t_additional[0], t_additional[-1]); ax.set_xlabel(r"$t$");
ax.set_title('Total Case Prediction')
ax.legend(['Data', 'Prediction'])
fig.show()
s
