import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

def save_history(path, hist):
    plt.figure(figsize=(15,7))

    plt.subplot(1,2,1)
    plt.plot(hist.history['acc'], label='train')
    plt.plot(hist.history['val_acc'], label='validation')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='validation')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(path + '/history.png', format='png', bbox_inches='tight')


def draw_predict(audio_name, genres, percent):
    # Draw plot
    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(1, 1, 1) 
    N = int(len(genres))
    index = np.arange(N)
    width = 0.35
    plt.bar(index, percent, width, label="percent", color="#87CEFA")
    for idx, num in enumerate(percent):
        plt.text(idx, num, '%1.3f'%(100*num) + '%', ha='center', va='bottom')
    plt.xlabel('genres')
    plt.ylabel('percent')

    plt.title('Prediction of %s'%audio_name)
    plt.xticks(index, genres)
    plt.yticks(np.arange(0, 1, 0.1))
    plt.legend(loc="upper right") 

    def to_percent(temp, position):
        return '%1.0f'%(100*temp) + '%'
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

    plt.savefig('../pic/predict/%s.png'%audio_name)
    plt.show()