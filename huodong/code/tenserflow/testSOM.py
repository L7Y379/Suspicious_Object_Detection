import math
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def TrainSom(source_data, source_label, target_data, target_label):
    source_data = np.array(source_data[:, 0:54000:270])
    target_data = np.array(target_data[:, 0:54000:270])
    source_label = np.argmax(source_label, 1)
    N = source_data.shape[0]
    M = source_data.shape[1]
    size = math.ceil(np.sqrt(5 * np.sqrt(N)))  # 经验公式
    max_iter = 2000
    som = MiniSom(size, size, M, sigma=3, learning_rate=0.1, neighborhood_function='bubble')
    som.pca_weights_init(source_data)
    som.train_batch(source_data, max_iter, verbose=False)
    winmap = som.labels_map(source_data, source_label)

    # 分类函数
    def classify(som_, data_, winmap_):
        from numpy import sum as npsum
        default_class = npsum(list(winmap_.values())).most_common()[0][0]
        result = []
        for d in data_:
            win_position = som_.winner(d)
            if win_position in winmap_:
                result.append(winmap_[win_position].most_common()[0][0])
            else:
                result.append(default_class)
        return result

    y_pred = classify(som, target_data, winmap)
    print(classification_report(np.argmax(target_label, 1), np.array(y_pred)))
    # 可视化
    label_name_map_number = {"up": 0, "down": 1, "walk": 2, "jump": 3}
    class_names = ["up", "down", "walk", "jump"]
    from matplotlib.gridspec import GridSpec
    plt.figure(figsize=(9, 9))
    the_grid = GridSpec(size, size)
    for position in winmap.keys():
        label_fracs = [winmap[position][label] for label in [0, 1, 2, 3]]
        plt.subplot(the_grid[position[1], position[0]], aspect=1)
        patches, texts = plt.pie(label_fracs)
        plt.text(position[0] / 100, position[1] / 100, str(len(list(winmap[position].elements()))),
                 color='black', fontdict={'weight': 'bold', 'size': 15},
                 va='center', ha='center')
    plt.legend(patches, class_names, loc='upper right', ncol=4)
    plt.show()

data = np.genfromtxt('datasets/different_people_gesture/420_gesture_data.csv', dtype=float, delimiter=',',
                         encoding='utf-8')
label = np.genfromtxt('datasets/different_people_gesture/420_gesture_label.csv', dtype=float, delimiter=',',
                          encoding='utf-8')
target_data = np.zeros([40, 54000])
source_data = np.zeros([160, 54000])
target_label = np.zeros([40, 4])
source_label = np.zeros([160, 4])
m,n = 0,0
for i in range(200):
    if i % 5 == 0:
        target_data[m, :] = data[i, :]
        target_label[m, :] = label[i, :]
        m = m + 1
    else:
        source_data[n, :] = data[i, :]
        source_label[n, :] = label[i, :]
        n = n + 1
TrainSom(source_data, source_label, target_data, target_label)