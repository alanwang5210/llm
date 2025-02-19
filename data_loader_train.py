from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import random

# torchvision用于处理图像数据集，可以对图像进行规范化、调整大小和裁剪
# ToTensor()方法能够将原始的PILImage格式或者numpy.array格式的数据
# 格式化为可被PyTorch快速处理的张量类型，能够把图片的灰度范围从0～255变换到0～1
transform = transforms.Compose([transforms.ToTensor()])

# 参数batch_size指定每个批次(batch)中应包含的样本数量，这里每个批次包含64个样本；
# 参数shuffle指定DataLoader在每个训练轮次(epoch)开始时随机打乱数据集中的样本顺序。
# 由于模型不会在每个轮次中都以相同的顺序看到样本，因此这种方式有助于改善模型的泛化能力。
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 使用iter()方法获取数据集中的所有图像
dataiter = iter(trainloader)
images, labels = next(dataiter)
print(images.shape)
print(labels.shape)
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')


class SampleDataset(Dataset):
    def __init__(self, r1, r2):
        randomlist = []
        for i in range(120):
            n = random.randint(r1, r2)
            randomlist.append(n)
        self.samples = randomlist

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.samples[idx]
        elif isinstance(idx, int):
            if idx >= len(self.samples):
                raise IndexError(f"Index {idx} is out of bounds for samples of length {len(self.samples)}")
            return self.samples[idx]
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

# 确保这段代码只在主模块中运行
if __name__ == '__main__':
    # 创建数据集实例
    dataset = SampleDataset(1, 100)

    # 使用DataLoader加载数据
    loader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=2)  # 设置num_workers为2
    for i, batch in enumerate(loader):
        print(i, batch)
