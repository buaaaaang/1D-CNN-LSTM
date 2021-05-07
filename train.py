from keras.utils.io_utils import HDF5Matrix
import kerastuner as kt
import model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

path = "C:/Users/LG/Desktop/skt/KoreanSpeechDataForSER"

batch_size = 32
maxepoch = 5

def train(model,x_tr,y_tr,x_v,y_v):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
    mc = ModelCheckpoint('model.h5', monitor='val_categorical_accuracy', mode='max', verbose=1, save_best_only=True)
    history = model.fit(x_tr,y_tr,epochs=maxepoch,batch_size=batch_size,validation_data=(x_v,y_v),callbacks=[es, mc])
    return model, history

def test(x_t,y_t):
	saved_model = load_model('model.h5')
	score = saved_model.evaluate(x_t,y_t,batch_size=batch_size)
	print(score)
	return score

if __name__ == "__main__":
    import IPython
    import h5py
    import tensorflow as tf
    import numpy as np
    from numpy import newaxis
    import matplotlib.pyplot as plt

    file = "C:/Users/LG/Desktop/skt/KoreanSpeechDataForSER/1D.hdf5"
    hf = h5py.File(file,'r')

    x_tr = (hf['x'][:16000])[:,:,newaxis]
    x_v = (hf['x'][16000:20000])[:,:,newaxis]
    x_t = (hf['x'][20000:])[:,:,newaxis]

    Y = hf['y']
    y = np.zeros((Y.size,5))
    y[np.arange(Y.size),Y] = 1.0
    y_tr = y[:16000]
    y_v = y[16000:20000]
    y_t = y[20000:]


    model = model.model()
    print(model.summary())

    model, hist = train(model,x_tr,y_tr,x_v,y_v)

    fig, loss_ax = plt.subplots()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper right')

    plt.savefig('loss-epoch.png')

    fig, acc_ax = plt.subplots()

    acc_ax.plot(hist.history['categorical_accuracy'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_categorical_accuracy'], 'g', label='val acc')
    acc_ax.set_xlabel('epoch')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper left')

    plt.savefig('acc-epoch.png')

    score = test(x_t,y_t)