import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def build_cnn(state_size, action_size, learning_rate):
    model = Sequential()
    model.add(Dense(24, input_dim = state_size, activation = 'relu'))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(action_size, activation = 'linear'))
    model.compile(loss = 'mse', optimizer = Adam(lr = learning_rate))
    return model
