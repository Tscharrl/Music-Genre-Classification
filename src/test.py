from datetime import datetime

from dataset_process import get_train_test_set
from model_cnn import *
from model_crnn import *
from train import *
from visualization import save_history

# get model
# model = get_cnn_model()
model = get_cnn_model()
print_model(model)
