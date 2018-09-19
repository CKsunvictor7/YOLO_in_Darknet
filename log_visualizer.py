import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
#%matplotlib inline


#    training_log/exp14_2_training_log.txt  ,training_log_exp13.txt
def draw_training_log():
    avg_loss = []
    skip_iteration = 0  # 5000
    with open('training_log/exp16_training_log.txt', 'r') as r:
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


    # set name of axis
    plt.xlabel("mini-batch", fontsize=13, fontweight='bold')
    plt.ylabel("avg_loss", fontsize=13, fontweight='bold')

    # see the detail part
    #y_range = (0, 1.5)
    #y_tick_max = 1.5

    # see overall
    y_range = (0, 1.5)
    y_tick_max = 1.5

    # 设置坐标轴范围
    # plt.xlim((-5, 5))  # 也可写成plt.xlim(-5, 5)
    plt.ylim(y_range)  # 也可写成plt.ylim(-2, 2)
    # 设置坐标轴刻度
    my_y_ticks = np.arange(0, y_tick_max, 0.05)

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