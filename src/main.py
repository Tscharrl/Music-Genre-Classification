from datetime import datetime

from dataset_process import get_train_test_set
from model_cnn import *
from model_crnn import *
from train import *
from visualization import save_history

# get train set & test set
train_input, train_labels, test_input, test_labels, train_size = get_train_test_set(type='both')

# get model
# model = get_cnn_model()
model = get_crnn_model()
print_model(model)

epo = [50]
bat = [64]
for j in range(len(bat)):
    for i in range(len(epo)):
        # Create directory to save logs
        exec_time = datetime.now().strftime('%Y%m%d%H%M%S')
        try:
            os.mkdir('../log/{}{}{}{}{}'.format(exec_time, ' epo=', epo[i], ' bat=', bat[j]))
        except FileExistsError:
            # If the directory already exists
            pass

        # train
        history = train(epo[i], bat[j], '../log/{}{}{}{}{}'.format(exec_time, ' epo=', epo[i], ' bat=', bat[j]), exec_time, train_input, train_labels, test_input, test_labels, train_size)

        save_history('../log/{}{}{}{}{}'.format(exec_time, ' epo=', epo[i], ' bat=', bat[j]), history)
