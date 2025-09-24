import math
import matplotlib.pyplot as plt
# # 绘制Epoch级图像函数
def report_plot_epoch(num_epochs, report_epoch):
    # 更新训练损失、训练准确度、测试损失、测试准确度图像
    # 转置一下report列表
    report_epoch_t = list(map(list, zip(*report_epoch)))

    plt.close('all')

    # 绘制第一张子图
    loss_plot = plt.subplot(2, 1, 1)
    loss_plot.plot(report_epoch_t[0], report_epoch_t[1],
                   color='tab:blue', label='train_loss')
    loss_plot.plot(report_epoch_t[0], report_epoch_t[3],
                   color='tab:orange', label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(1, num_epochs)
    # plt.ylim(0, 1)
    plt.xticks(range(1, num_epochs + 1, math.ceil(num_epochs / 10)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid()

    # 绘制第二张子图
    acc_plot = plt.subplot(2, 1, 2)
    acc_plot.plot(report_epoch_t[0], report_epoch_t[2],
                  color='tab:blue', label='train_acc')
    acc_plot.plot(report_epoch_t[0], report_epoch_t[4],
                  color='tab:orange', label='test_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.xlim(1, num_epochs)
    plt.ylim(0, 1)
    plt.xticks(range(1, num_epochs + 1, math.ceil(num_epochs / 10)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid()

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.pause(0.1)  # 图片停留0.1s


# # 绘制Batch级图像函数
def report_plot_batch(num_epochs, report_batch, report_epoch):
    # 更新训练损失、训练准确度、测试损失、测试准确度图像
    # 转置一下report列表
    report_batch_t = list(map(list, zip(*report_batch)))
    report_epoch_t = list(map(list, zip(*report_epoch)))

    plt.close('all')

    # 绘制第一张子图
    loss_plot = plt.subplot(2, 1, 1)
    loss_plot.plot(report_batch_t[0], report_batch_t[1],
                   color='tab:blue', label='train_loss')
    if not report_epoch_t == []:
        loss_plot.plot(report_epoch_t[0], report_epoch_t[3],
                       color='tab:orange', label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(0, num_epochs)
    # plt.ylim(0, 1)
    plt.xticks(range(0, num_epochs + 1, math.ceil(num_epochs / 10)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid()

    # 绘制第二张子图
    acc_plot = plt.subplot(2, 1, 2)
    acc_plot.plot(report_batch_t[0], report_batch_t[2],
                  color='tab:blue', label='train_acc')
    if not report_epoch_t == []:
        acc_plot.plot(report_epoch_t[0], report_epoch_t[4],
                      color='tab:orange', label='test_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.xlim(0, num_epochs)
    plt.ylim(0, 1)
    plt.xticks(range(0, num_epochs + 1, math.ceil(num_epochs / 10)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid()

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.pause(0.1)  # 图片停留0.1s