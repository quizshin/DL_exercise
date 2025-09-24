import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# 数据集图片标签文件路径
img_label_file_dir = r'C:\Users\15020\Desktop\leaves\train.csv'  # 修改为您的标签文件路径
# 读取数据集图片标签文件
img_label = pd.read_csv(img_label_file_dir)

# 按标签列分层抽样
train_img_label, test_img_label = train_test_split(img_label, test_size=0.2, stratify=img_label.iloc[:, 1])

# 打印划分结果
print('\n训练集标签分布：\n', train_img_label['label'].value_counts())
print('\n测试集标签分布：\n', test_img_label['label'].value_counts())

# 将划分好的数据写入相应文件
train_img_label.to_csv("split_train.csv", index=False)
test_img_label.to_csv("split_test.csv", index=False)

# 按照划分结果将图片复制到符合 ImageFolder 格式的目录中
def split_image(img_label, output_folder, root_path):
    """
    根据划分结果将图片复制到符合 ImageFolder 格式的目录中。

    Args:
        img_label (DataFrame): 包含图片路径和标签的 DataFrame。
        output_folder (str): 输出文件夹路径。
        root_path (str): 原始图片的根目录。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, row in img_label.iterrows():
        img_name, label = row[0], row[1]
        print(img_name, label)
        label_folder = os.path.join(output_folder, label)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)

        # 修复路径拼接问题
        src_path = os.path.join(root_path, os.path.basename(img_name))  # 使用完整路径拼接
        dst_path = os.path.join(label_folder, os.path.basename(img_name))  # 目标路径只包含文件名

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"警告：文件 {src_path} 不存在，跳过复制。")

# 定义原始图片路径和输出路径
root_path = r'C:/Users/15020/Desktop/leaves/images'  # 原始图片的根目录
train_output_folder = r'C:/Users/15020/Desktop/leaves/train_images'  # 训练集目标路径
test_output_folder = r'C:/Users/15020/Desktop/leaves/test_images'  # 测试集目标路径

# 复制训练集和测试集图片
split_image(train_img_label, train_output_folder, root_path)
split_image(test_img_label, test_output_folder, root_path)

print("图片划分完成！")