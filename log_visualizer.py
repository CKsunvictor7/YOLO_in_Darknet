import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
#%matplotlib inline



def draw_training_log():
    avg_loss = []
    skip_iteration = 0  # 5000
    with open('training_log_UEC_test.txt', 'r') as r:
        for c, line in enumerate(r.readlines()):
            if ' avg' in line:
                if skip_iteration == 0:
                    pieces = line.split(' ')
                    loss = float(pieces[3])
                    avg_loss.append(loss)
                else:
                    skip_iteration = skip_iteration - 1
    plt.figure(figsize=(8, 5))
    plt.plot(avg_loss, label='avg_loss')
    # 设置坐标轴范围
    # plt.xlim((-5, 5))  # 也可写成plt.xlim(-5, 5)
    plt.ylim((0, 7))  # 也可写成plt.ylim(-2, 2)

    # set name of axis
    plt.xlabel("mini-batch", fontsize=13, fontweight='bold')
    plt.ylabel("avg_loss", fontsize=13, fontweight='bold')
    # 设置坐标轴刻度
    #my_x_ticks = np.arange(-5, 5, 0.5)
    my_y_ticks = np.arange(0, 5, 0.5)
    #plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    """
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(avg_loss, label='avg_loss')
    ax.legend(loc='best')
    ax.set_title('The loss curves')
    ax.set_xlabel('images')
    my_y_ticks = np.arange(0, 0.5, 5)
    plt.yticks(my_y_ticks)
    """
    plt.show()


def main():
    draw_training_log()


if __name__ == '__main__':
    main()