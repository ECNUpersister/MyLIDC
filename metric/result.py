import matplotlib.pyplot as plt
import pandas as pd


def plot_loss(df, dir, title):
    plt.plot(df.epoch, df.train_loss, marker='o', color='r')
    plt.plot(df.epoch, df.val_loss, marker='+', color='b')
    plt.title(title, fontsize=13)
    plt.ylabel('loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=14)
    plt.legend(['train', 'val'], fontsize=12, loc='best')
    plt.savefig(dir + title + '.png', dpi=300)
    plt.show()


def plot_metric(df, dir, title, iou=False):
    # 计算dice和iou
    if not iou:
        plt.plot(df.epoch, df.train_dice, marker='x', color='r')
        plt.plot(df.epoch, df.val_dice, marker='+', color='orangered')
    else:
        plt.plot(df.epoch, df.train_iou, marker='o', color='navy')
        plt.plot(df.epoch, df.val_iou, marker='+', color='royalblue')
    plt.title(title, fontsize=13)
    score = 'iou' if iou else 'dice'
    plt.ylabel(score, fontsize=14)
    plt.xlabel('Epochs', fontsize=14)
    plt.legend(['train', 'val'], fontsize=12, loc='best')
    plt.savefig(dir + title + '.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    file_name = '2021-05-19_01_00_51'
    model_result = pd.read_csv('../model_outputs/{}/log.csv'.format(file_name))
    plot_metric(model_result, '../model_outputs/{}/'.format(file_name), 'DICE')
    plot_metric(model_result, '../model_outputs/{}/'.format(file_name), 'IOU', True)
    plot_loss(model_result, '../model_outputs/{}/'.format(file_name), 'LOSS')
