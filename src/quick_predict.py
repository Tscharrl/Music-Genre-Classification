import numpy as np
from keras.models import load_model
from prettytable import PrettyTable

from preprocessing import *
from visualization import draw_predict

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# get example
audio_name = input("Input audio name: ")
path = '../example/' + audio_name
features = get_features(path)
eg_input = np.array(features).reshape(1, 20, 1250)
eg_input = np.expand_dims(eg_input, axis=3)

# get model
model = load_model('../model/CNN.h5')

# predict
predict = model.predict(eg_input)
predict = predict[0]

# Transfer output to percent
percent = []
summ = np.sum(predict)
for i in range(len(predict)):
    percent.append(predict[i]/summ)

# Print prediction table
print('      Prediction      ')
pred = PrettyTable(['genre', 'percent'])
for i in range(len(genres)):
    # pred.add_row([genres[i], str('%1.3f'%(100*percent[i]) + '%')])
    pred.add_row([genres[i], float('%1.3f'%percent[i])])
pred.sortby = 'percent'
pred.reversesort = True
print(pred)

draw_predict(audio_name, genres, percent)