import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import time

#reading the csv file
filepath = "/Users/student/cs/microsoft/microsoft-internship/data/MSFT_2006-01-01_to_2018-01-01.csv"
data = pd.read_csv(filepath)
data = data.sort_values('Date') 
data.head()

#drawing the graph
sns.set_style("darkgrid")
plt.figure(figsize = (15,9))
plt.plot(data[['Close']])
plt.xticks(range(0,data.shape[0],500),data['Date'].loc[::500],rotation=45) #date var
plt.title("Microsoft Stock Price",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price (USD)',fontsize=18)


#data normalization

price = data[['Close']]


scaler = MinMaxScaler(feature_range=(-1, 1))
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))


'''
Split Data

Input:
    []
'''
def split_data(stock, lookback):

    data_raw = stock.to_numpy()
    #1d array soon to become a 2d array with different parts of our raw data
    data = []
    
    #raw data traversal to create the 2d data array with the data pockets
    #lookback: determins the size of the data pockets to be used
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    #turn the data array into a np array to utilize np functions
    data = np.array(data)
    
    #using a 20% test size and a 80% training size for the data
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)


    """
    slicing our 3 dimensional data array into 3 categories:
        1: First dimension is cut from 0 to :train_set_size, which is 80% of the data
        2: Size of the data pocket without the last point in x, and in y only the last data point so that the model can
           learn if it reached the goal
        3: number of columns used, ie if it's only the "close" value it is 1. 
    """

    x_train = data[:train_set_size,:-1, :]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]

    return [x_train, y_train, x_test, y_test]


#calling the above function "split_data" 
#choose lookback size
lookback = 100
x_train, y_train, x_test, y_test = split_data(price, lookback)

"""
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)
"""

#tensors defined
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)


#y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
#y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

#constants for the neural network
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100

#defining the neural network
class LSTM(nn.Module):
    

    """
    Constructor for the lstm nn
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

#the model object 
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr = 0.01)




#12
hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []

for t in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train_lstm)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))
##

#13
predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))

import seaborn as sns
sns.set_style("darkgrid")    

fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)

plt.subplot(1, 2, 1)
ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (LSTM)", color='tomato')
ax.set_title('Stock price', size = 14, fontweight='bold')
ax.set_xlabel("Days", size = 14)
ax.set_ylabel("Cost (USD)", size = 14)
ax.set_xticklabels('', size=10)


plt.subplot(1, 2, 2)
ax = sns.lineplot(data=hist, color='royalblue')
ax.set_xlabel("Epoch", size = 14)
ax.set_ylabel("Loss", size = 14)
ax.set_title("Training Loss", size = 14, fontweight='bold')
fig.set_figheight(6)
fig.set_figwidth(16)
plt.show()



