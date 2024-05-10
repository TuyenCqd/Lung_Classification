
from VggNet import VGGNet
from train_VggNet import get_args
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_args():
    parser = argparse.ArgumentParser(description="Train an CNN model")
    parser.add_argument("--checkpoint_path", "-m", type=str, default="lung_checkpoints/model.pt")
    args = parser.parse_args()
    return args

def visualize_data(train_path, val_path, test_path):
    # Tạo một DataFrame mô phỏng dữ liệu

    train_NORMAL = len(os.listdir(os.path.join(train_path, 'NORMAL')))
    train_PNEUMONIA = len(os.listdir(os.path.join(train_path, 'PNEUMONIA')))
    val_NORMAL = len(os.listdir(os.path.join(val_path, 'NORMAL')))
    val_PNEUMONIA = len(os.listdir(os.path.join(val_path, 'PNEUMONIA')))
    test_NORMAL = len(os.listdir(os.path.join(test_path, 'NORMAL')))
    test_PNEUMONIA = len(os.listdir(os.path.join(test_path, 'PNEUMONIA')))

    data = {
        'gender': ['Train', 'Validation', 'Test'],
        'NORMAL': [train_NORMAL, val_NORMAL, test_NORMAL],
        'PNEUMONIA': [train_PNEUMONIA, val_PNEUMONIA, test_PNEUMONIA]
    }
    df = pd.DataFrame(data)

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df['gender']))
    width = 0.4
    rects1 = ax.bar(x - width / 2, df['NORMAL'], width, label='NORMAL')
    rects2 = ax.bar(x + width / 2, df['PNEUMONIA'], width, label='PNEUMONIA')

    # Thêm số lượng mẫu ở trên mỗi cột
    for i, v in enumerate(df['NORMAL']):
        ax.text(i - width / 2, v + 10, str(int(v)), ha='center')
    for i, v in enumerate(df['PNEUMONIA']):
        ax.text(i + width / 2, v + 10, str(int(v)), ha='center')

    ax.set_xlabel('Gender')
    ax.set_ylabel('Số lượng mẫu', fontdict={'weight': 'bold'})
    ax.set_title('Số lượng mẫu "NORMAL" và "PNEUMONIA" trong các tập dữ liệu')
    ax.set_xticks(x)
    ax.set_xticklabels(df['gender'])
    ax.legend()
    plt.show()

def visualize_model(checkpoint_path):
    args = get_args()
    checkpoint = torch.load(args.checkpoint_path)
    train_loss = checkpoint["train_loss"]
    train_acc = checkpoint["train_acc"]
    val_loss = checkpoint["val_loss"]
    val_acc = checkpoint["val_acc"]

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 6))

    # Vẽ đường training loss
    # ax.plot(train_loss, label='Training Loss')
    ax.plot(train_acc, label='Training Acc')

    # Vẽ đường validation loss
    # ax.plot(val_loss, label='Validation Loss')
    ax.plot(val_acc, label='Validation Acc')

    # Thêm các nhãn
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training và Validation Acc')
    ax.legend()

    # Hiển thị biểu đồ
    plt.show()

if __name__ == '__main__':
    # train_path = 'data/lung_dataset/train'
    # val_path = 'data/lung_dataset/val'
    # test_path = 'data/test_lung_dataset'
    # visualize_data(train_path, val_path, test_path)
    visualize_model('lung_checkpoints/model.pt')