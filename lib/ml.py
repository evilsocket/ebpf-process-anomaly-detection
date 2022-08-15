# 
# Copyleft Simone 'evilsocket' Margaritelli
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

from lib import MAX_SYSCALLS


class AutoEncoder(object):
    def __init__(self, filename, load=False):
        self.filename = filename
        self.model = None
        if load:
            from tensorflow.keras.models import load_model
            print("loading model from %s ..." % filename)
            self.model = load_model(filename)


    def create(n_inputs = MAX_SYSCALLS):
        from tensorflow.keras import Input, Model
        from tensorflow.keras.layers import Dense, ReLU

        # define the autoencoder model
        inp = Input(shape=(n_inputs,))
        encoder = Dense(n_inputs)(inp)
        encoder = ReLU()(encoder)
        middle = Dense(int(n_inputs / 2))(encoder)
        decoder = Dense(n_inputs)(middle)
        decoder = ReLU()(decoder)
        decoder = Dense(n_inputs, activation='linear')(decoder)
        m = Model(inp, decoder)
        m.compile(optimizer='adam', loss='mse')
        
        return m

    def train(self, datafile, epochs, batch_size, test_size=0.1):
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split

        # load data and drop columns we don't need for training
        train_df = pd.read_csv(datafile)
        train_val_df = train_df.drop(['sample_time'], axis=1)
        train = train_val_df.values
        # split into train and test datasets
        train, test = train_test_split(train, test_size=test_size)
        # start training
        self.model = AutoEncoder.create()
        self.model.fit(train, train, validation_data=(test, test), epochs=epochs, batch_size=batch_size, verbose=1)
        # save to file
        self.model.save(self.filename)

        print("model saved to %s, getting error threshold for %d samples ..." % (self.filename, len(test)))

        # test the model on test data to calculate the error threshold
        y_test = self.model.predict(test)
        test_err = []

        for ind in range(len(test)):
            abs_err = np.abs(test[ind, :]-y_test[ind, :])
            test_err.append(abs_err.sum())

        threshold = max(test_err)

        return self.model, threshold
    
    def predict(self, X):
        import numpy as np

        # make sure we have a numpy array as input
        X = np.asarray(X, dtype=np.float32)
        # reconstruct from input
        y = self.model.predict(X, verbose=0)
        # get errors vector
        err = np.abs(X[0] - y[0])
        
        return (y, err, sum(err))
