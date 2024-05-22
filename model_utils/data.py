from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
from sklearn.model_selection import train_test_split  # 把一个数据集划分为训练集和验证集
from tqdm import tqdm
import numpy as np



def read_txt_data(path):
    label = []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f)):
            if i == 0:
                continue  # continue 表示立即执行下一次循环

            line = line.strip('\n')  # 删除结尾的换行符
            line = line.split(",", 1)  # 在 ， 处分割，1表示分割次数
            label.append(line[0])
            data.append(line[1])

    print(len(label))
    return data, label


class JdDataset(Dataset):
    def __init__(self, x, label):
        self.X = x
        label = [int(i) for i in label]
        self.Y = torch.LongTensor(label)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]            # 数据集一般不让返回str， 要写在字典中，或者转为矩阵。

    def __len__(self):
        return len(self.Y)


def get_dataloader(path, batchsize=1, valSize=0.2):
    x, label = read_txt_data(path)
    train_x, val_x, train_y, val_y = train_test_split(x, label, test_size=valSize, shuffle=True, stratify=label)  # stratify=label 按标签的比例取数据
    train_set = JdDataset(train_x, train_y)
    val_set = JdDataset(val_x, val_y)
    train_loader = DataLoader(train_set, batch_size=batchsize)
    val_loader = DataLoader(val_set, batch_size=batchsize)
    return train_loader, val_loader



if __name__ == '__main__':

    get_dataloader("../jiudian.txt", batchsize=16)




