# importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder  


# importing the dataset
dataset = pd.read_csv('dataset.csv')
cols_to_transform = ['Flags']
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X = pd.get_dummies( X, columns = cols_to_transform )
data = np.asarray(X)

le = LabelEncoder()
le.fit(y)
y = le.transform(y)
#y[y == 2] = 0
#df_with_dummies.iloc[:,2][df_with_dummies.iloc[:,2]==2]=0



# scaling the values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# training the som
from minisom import MiniSom
som = MiniSom(x = 20, y = 20, input_len = 56, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 30)

# visualising the data
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['s','o','s']
colors = ['g','r','g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# detecting the frauds
mappings = som.win_map(X)
# write in form of (row,column) = (x-axis, y-axis)
frauds = np.concatenate((mappings[(5,9)], mappings[(7,6)]), axis = 0)
frauds = sc.inverse_transform(frauds)

