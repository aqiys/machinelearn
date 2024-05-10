### 处理数据方法：Torch.utils.data.DataLoader和Torch.utils.data.Dataset

Dataset存储了样本及其相应的标签

·····

自定义Dataset需要继承torch.utils.data.Dataset并实现两个成员方法：

1. `__getitem__()`

决定了每次如何读取数据，以图片为例：

```python
 def __getitem__(self, index):
        img_path, label = self.data[index].img_path, self.data[index].label
        img = Image.open(img_path)

        return img, label
```

torchvision.transforms中，常用transform有Resize，RandomCrop，Normalize，ToTensor

(ToTensor很重要，可把一个PIL图片或numpy图片转为torch.Tensor,建议在`__getitem__()`里面用PIL来读图片, 而不是用skimage.io)

2. `__len__()`

	返回数据集长度

	```python
	  def __len__(self):
	        return len(self.data)
	```

·····

DataLoader则围绕Dataset包装了一个-==可迭代==的数据。

在实际项目中，如果数据量很大，考虑到内存有限、I/O 速度等问题，在训练过程中不可能一次性的将所有数据全部加载到内存中，也不能只用一个进程去加载，需要多进程、迭代加载，**DataLoader** 是一个迭代器，对其传入一个 **Dataset 对象**，它会根据参数 **batch_size** 的值生成一个 batch 的数据，节省内存的同时，它还可以实现多进程、数据打乱等处理。

类定义为

```python
class torch.utils.data.DataLoader(
dataset, 
batch_size=1, 
shuffle=False, 
sampler=None, 
batch_sampler=None, 
num_workers=0,  # 是否多线程
collate_fn=<function default_collate>,  # 用来打包batch
pin_memory=False, 
drop_last=False
)
```

DataLoader常用参数：

- dataset表示Dataset类，决定数据从哪里读取，如何读取
- batch_size表示批大小
- num_works表示是否多进程读取数据
- shuffle表示每个epoch是否乱序
- drop_last表示当样本数不能被batch_size整除时，是否舍弃最后一批数据

·····

（自定义数据集后）在for 循环里, 总共有三点操作:

1. 调用了`dataloader` 的`__iter__() `方法, 产生了一个`DataLoaderIter`

2. 反复调用`DataLoaderIter` 的`__next__()`来得到batch, 具体操作就是, 多次调用dataset的`__getitem__()`方法 (如果`num_worker`>0就多线程调用), 然后用`collate_fn`来把它们打包成batch. 中间还会涉及到`shuffle` , 以及`sample` 的方法等, 这里就不多说了.

3. 当数据读完后,` __next__()`抛出一个`StopIteration`异常, `for`循环结束, `dataloader` 失效.

	DataLoader数据读取机制：

	![img](https://pic2.zhimg.com/80/v2-40815d562de5d4cd724a7e186fe12865_720w.webp)

·····

pytorch提供特定领域且包含数据集的库，如TorchText、TorchVision和TorchAudio

使用一个TorchVision数据集，

torchvision.datasets模块包含众多数据集对象，如CIFAR、COCO

TorchVision数据集包括两个参数：transform和target_transform，分别用来修改样本和标签

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
```

将数据集做参数传递至DataLoader，支持自动批处理、采样、洗牌和多进程数据加载等

```python
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)		#第一参数即数据集，下同
test_dataloader = DataLoader(test_data, batch_size=batch_size)
```

### 搭建层

创建继承自nn.Module的类。在`__init__`函数定义层，在forward中排列组合

有GPU可以转移至GPU

```python
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

输出：

```python
Using cuda device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

### 损失函数，优化器

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

在一个单一的训练循环中，模型对训练数据集（分批送入）进行预测，并通过反向传播预测误差来调整模型的参数。

```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

根据测试集检查性能，确保模型参数在学习

```
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")1
```

训练过程是通过几个迭代（epochs）进行的。在每个epoch中，模型学习参数以做出更好的预测。我们在每个epoch中打印模型的准确度和损失；我们希望看到准确度在每个epoch中增加，损失在每个epoch中减少。

```
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

### 保存模型

序列化内部状态字典（包括模型参数）

```python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

### 载入模型

重新创建模型结构并将状态字典加载到其中

```python
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
```

用该模型进行预测

```python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

