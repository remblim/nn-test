import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import time
#
Y = np.array([[0,1,1],[0,1,1],[1,0,1],[1,0,1],[0,1,0],[1,1,1]]).reshape(-1,3)
X = np.array([[1,1,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1]]).reshape(-1,3,1)
print(X,Y)

network = input_data(shape = [None, 3, 1], name='input')
network = fully_connected(network, 1000, activation='tanh')
network = fully_connected(network, 1000, activation='tanh')
network = fully_connected(network, 1000, activation='tanh')
network = fully_connected(network, 3, activation='relu')
network = regression(network,optimizer='adam', learning_rate = 1e-3, loss='mean_square', name='targets')
model = tflearn.DNN(network, tensorboard_dir='log')
predict = model.predict(X)
print(predict)
model.fit(X,Y, n_epoch=10000, snapshot_step=1000, show_metric=True, run_id='openaistuff')
predict = model.predict(X)
print(predict)