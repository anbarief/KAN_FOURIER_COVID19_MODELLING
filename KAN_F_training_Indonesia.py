#AUTHOR: ARIEF ANBIYA
#EMAIL: ariefanbiya@gmail.com
#DATE CREATED: 9-7-2026

import torch
import torch.nn as nn
import torch.optim as optim
import pandas 
import numpy as np
import matplotlib.pyplot as plt


n_epoch = 5000; learning_rate = 0.00002

all_t = pandas.read_csv('index_ready_INA.csv')['t']
dt = 1
Dt = torch.tensor(dt) 
index_start = int(365*(1/dt)) #1 Mar 2021
index_end = int(671*(1/dt)) #1 Jan 2022
N_p = 274*(10**6)    #Initial population at t=0 (1 Mar 2020), using estimates of 1 January 2020 population (WPP UN data)
t_data = np.array(list(all_t)[index_start:index_end+1])
n = len(t_data)
torch.manual_seed(1);


#Setup and Normalize Epidemic Data
S_data = np.array(list(pandas.read_csv('S_ready_INA.csv')['S'][index_start:index_end+1]))/N_p
I_data = np.array(list(pandas.read_csv('I_ready_INA.csv')['I'][index_start:index_end+1]))/N_p
R_data = np.array(list(pandas.read_csv('R_ready_INA.csv')['R'][index_start:index_end+1]))/N_p
D_data = np.array(list(pandas.read_csv('D_ready_INA.csv')['D'][index_start:index_end+1]))/N_p

diff_S = (S_data[1:]-S_data[0:-1])/dt
diff_I = (I_data[1:]-I_data[0:-1])/dt
diff_R = (R_data[1:]-R_data[0:-1])/dt
diff_D = (D_data[1:]-D_data[0:-1])/dt

mean_t = np.mean(t_data); std_t = np.std(t_data)
mean_S = np.mean(S_data); std_S = np.std(S_data)
mean_I = np.mean(I_data); std_I = np.std(I_data)
mean_R = np.mean(R_data); std_R = np.std(R_data)
mean_D = np.mean(D_data); std_D = np.std(D_data)

mean_S_diff = np.mean(diff_S); std_S_diff = np.std(diff_S)
mean_I_diff = np.mean(diff_I); std_I_diff = np.std(diff_I)
mean_R_diff = np.mean(diff_R); std_R_diff = np.std(diff_R)
mean_D_diff = np.mean(diff_D); std_D_diff = np.std(diff_D)

t_tensor = (t_data-mean_t)/std_t
S_tensor = (S_data-mean_S)/std_S
I_tensor = (I_data-mean_I)/std_I
R_tensor = (R_data-mean_R)/std_R
D_tensor = (D_data-mean_D)/std_D

diff_S_tensor = (diff_S-mean_S_diff)/std_S_diff
diff_I_tensor = (diff_I-mean_I_diff)/std_I_diff
diff_R_tensor = (diff_R-mean_R_diff)/std_R_diff
diff_D_tensor = (diff_D-mean_D_diff)/std_D_diff
### Ends here


### Setup Data to Feed the Input Layer of NN 
input_tensor = torch.tensor(np.array([[S_tensor[i], I_tensor[i], R_tensor[i], D_tensor[i], \
                                       diff_S_tensor[i], diff_I_tensor[i], diff_R_tensor[i], diff_D_tensor[i]] for i in range(n-1)], dtype=np.float32))
n_input_tensor = len(input_tensor)
### Ends here


S_data = torch.tensor(S_data, dtype=torch.float32).reshape(-1, 1)
I_data = torch.tensor(I_data, dtype=torch.float32).reshape(-1, 1)
R_data = torch.tensor(R_data, dtype=torch.float32).reshape(-1, 1)
D_data = torch.tensor(D_data, dtype=torch.float32).reshape(-1, 1)


### KAN-F
class KAN_Fourier(nn.Module):
    def __init__(self):
        super(KAN_Fourier, self).__init__()

        self.sigmoid = nn.Sigmoid()
        
        self.n_data = n_input_tensor
        self.F = 30
        self.n_coef_fourier = 2*self.F
        self.n_input_dim = 8
        self.layers_size = [self.n_input_dim, 2*(self.n_input_dim) + 1, 2*(2*(self.n_input_dim) + 1)+1, 1]

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
    

# Ends here

## Instantiate the 3 surrogate neural networks
beta_model = KAN_Fourier()
gamma_model = KAN_Fourier()
mu_model = KAN_Fourier()
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
def loss_function(S_data, I_data, R_data, D_data, \
                  beta_model_all, gamma_model_all, mu_model_all):
    RK4_result = RK4(S_data[0:-1], I_data[0:-1], R_data[0:-1], D_data[0:-1], \
                  beta_model_all, gamma_model_all, mu_model_all, Dt)
    
    loss_S = (1)*( torch.sqrt(torch.mean((S_data[1:] - RK4_result[0])**2)))    
    loss_I = (500)*( torch.sqrt(torch.mean((I_data[1:] - RK4_result[1])**2)))
    loss_R = 5*(100)*( torch.sqrt(torch.mean((R_data[1:] - RK4_result[2])**2)))    
    loss_D = 10*(2000)*( torch.sqrt(torch.mean((D_data[1:] - RK4_result[3])**2)))
    
    total_loss = (loss_S + loss_I + loss_R + loss_D) 

    return total_loss, loss_S, loss_I, loss_R, loss_D
# --------------- ENDS HERE ---------------------------------------------


## ------------------ FUNCTION FOR TRAINING THE NEURAL NETWORKS ---------------------------------------------------
def train_SIRD_model(max_epoch, beta_model, gamma_model, mu_model):

    optimizer = optim.Adam(list(beta_model.parameters()) + list(gamma_model.parameters()) + list(mu_model.parameters()), lr = learning_rate)
    total_loss_values = []
    log_loss_values = []
    log_loss_values_S = []
    log_loss_values_I = []
    log_loss_values_R = []
    log_loss_values_D = []
    

    for epoch in range(max_epoch):
        optimizer.zero_grad();

        features = input_tensor
        beta_model_all = beta_model(features);
        gamma_model_all = gamma_model(features);
        mu_model_all = mu_model(features);

        
            
        loss, loss_S, loss_I, loss_R, loss_D = loss_function(S_data, I_data, R_data, D_data, \
                                                             beta_model_all, gamma_model_all, mu_model_all)
        true_loss = loss.item()
        loss.backward()    
        optimizer.step()
        
        loss, loss_S, loss_I, loss_R, loss_D = loss_function(S_data, I_data, R_data, D_data, \
                                                            beta_model_all, gamma_model_all, mu_model_all)


        total_loss_values.append(loss.item())
        log_loss_values.append(np.log(loss.item()))
        log_loss_values_S.append(torch.log(loss_S).item())
        log_loss_values_I.append(torch.log(loss_I).item())
        log_loss_values_R.append(torch.log(loss_R).item())
        log_loss_values_D.append(torch.log(loss_D).item())

        if epoch % 20 == 0:
            print(f"{epoch}, S:{loss_S.item()}, I:{loss_I.item()}, R:{loss_R.item()}, D:{loss_D.item()}, all:{loss.item()}")


        if (loss.item() < 0.0005) :
            break

    print(f"S:{loss_S.item()},I:{loss_I.item()},R:{loss_R.item()},D:{loss_D.item()}, all:{loss.item()}")
    
    return total_loss_values, log_loss_values, log_loss_values_S, log_loss_values_I, log_loss_values_R, log_loss_values_D
#####--------------------------------------- FUNCTION END HERE ---------------------------------------


##---------------- TRAIN THE NEURAL NETWORK -------------------------
import time
start_tm = time.time()
loss_list, log_loss_list, log_loss_list_S, log_loss_list_I, log_loss_list_R, log_loss_list_D = train_SIRD_model(n_epoch, beta_model, gamma_model, mu_model)
end_tm = time.time()
print("elapsed time = " + str(end_tm-start_tm))
#----------------- TRAIN ENDS HERE -----------------

beta_model.eval(); gamma_model.eval(); mu_model.eval(); 
beta = beta_model(input_tensor).detach().numpy();
gamma = gamma_model(input_tensor).detach().numpy();
mu = mu_model(input_tensor).detach().numpy();


##Solve the SIRD numerically using the estimated rates obtained by training NN
S = [S_data[0].detach().numpy()]; I = [I_data[0].detach().numpy()]; R = [R_data[0].detach().numpy()]; D = [D_data[0].detach().numpy()];
for i in range(n-1):
    K1 = F(S[i],I[i],R[i],D[i], beta[i], gamma[i], mu[i], dt)
    K2 = F(S[i]+K1[0]/2, I[i]+K1[1]/2, R[i]+K1[2]/2, D[i]+K1[3]/2, beta[i], gamma[i], mu[i], dt)
    K3 = F(S[i]+K2[0]/2, I[i]+K2[1]/2, R[i]+K2[2]/2, D[i]+K2[3]/2, beta[i], gamma[i], mu[i], dt)
    K4 = F(S[i]+K3[0], I[i]+K3[1], R[i]+K3[2], D[i]+K3[3], beta[i], gamma[i], mu[i], dt)
    
    S.append( S[i] + (1/6)*(K1[0] + K4[0]) + (1/3)*(K2[0]+K3[0]) );
    I.append( I[i] + (1/6)*(K1[1] + K4[1]) + (1/3)*(K2[1]+K3[1]) );
    R.append( R[i] + (1/6)*(K1[2] + K4[2]) + (1/3)*(K2[2]+K3[2]) );
    D.append( D[i] + (1/6)*(K1[3] + K4[3]) + (1/3)*(K2[3]+K3[3]) );

##Plot the Model Fitting
fig,ax = plt.subplots(2,2);

ax[0][0].plot(t_data, N_p*S_data.detach().numpy(), 'o', color='orange', lw=1, ms=3);ax[0][0].plot(t_data, np.array(S)*N_p,'-',color='blue',lw=1); ax[0][0].set_xlim(t_data[0], t_data[-1]); ax[0][0].set_xlabel(r"$t$"); ax[0][0].set_title(r"$S(t)$ v.s. $S_{model}(t)$", fontsize=12);

ax[0][1].plot(t_data, N_p*I_data.detach().numpy(), 'o', color='orange', lw=1, ms=3); ax[0][1].plot(t_data, np.array(I)*N_p,'-',color='blue',lw=1); ax[0][1].set_xlim(t_data[0], t_data[-1]); ax[0][1].set_xlabel(r"$t$"); ax[0][1].set_title(r"$I(t)$ v.s. $I_{model}(t)$", fontsize=12);

ax[1][0].plot(t_data, N_p*D_data.detach().numpy(), 'o', color='orange', lw=1, ms=3); ax[1][0].plot(t_data, np.array(D)*N_p,'-',color='blue',lw=1);
ax[1][0].set_xlim(t_data[0], t_data[-1]); ax[1][0].set_xlabel(r"$t$"); ax[1][0].set_title(r"$D(t)$ v.s. $D_{model}(t)$", fontsize=12);

ax[1][1].plot(t_data, N_p*R_data.detach().numpy(), 'o', color='orange', lw=1, ms=3); ax[1][1].plot(t_data, np.array(R)*N_p,'-',color='blue',lw=1);
ax[1][1].set_xlim(t_data[0], t_data[-1]); ax[1][1].set_xlabel(r"$t$"); ax[1][1].set_title(r"$R(t)$ v.s. $R_{model}(t)$", fontsize=12);

fig.tight_layout(pad=0.5); fig.show();
##
##
#####Plot the Estimated Rates
fig,ax=plt.subplots(1,3,figsize=(9,3));
ax[0].plot(t_data[0:-1], beta/N_p, color='blue',lw=3,alpha=0.3); ax[0].set_title(r"$\beta_{\Delta t}/N_{P}$", fontweight='heavy');
ax[1].plot(t_data[0:-1], gamma, color='blue',lw=3,alpha=0.3); ax[1].set_title(r"$\gamma_{\Delta t}$", fontweight='heavy');
ax[2].plot(t_data[0:-1], mu, color='blue',lw=3,alpha=0.3); ax[2].set_title(r"$\mu_{\Delta t}$", fontweight='heavy');
 
### Moving average
nt = len(t_data[0:-1]);
beta_avg = [ np.mean(beta[i-7:i])/N_p for i in range(7,nt+1) ];
gamma_avg = [ np.mean(gamma[i-7:i]) for i in range(7,nt+1) ];
mu_avg = [ np.mean(mu[i-7:i]) for i in range(7,nt+1) ];
ax[0].plot(t_data[6:-1], beta_avg, color='red',lw=1);
ax[1].plot(t_data[6:-1], gamma_avg, color='red',lw=1);
ax[2].plot(t_data[6:-1], mu_avg, color='red',lw=1);
fig.tight_layout(pad=0.5); fig.show();
##
#### Plot the Loss Functions
fig,ax=plt.subplots();
ax.plot(log_loss_list_S);
ax.plot(log_loss_list_I);
ax.plot(log_loss_list_R);
ax.plot(log_loss_list_D);
fig.show()
