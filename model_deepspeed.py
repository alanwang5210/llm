import deepspeed
import torch
import torch.nn as nn
from torch.optim import Adam


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


# 初始化模型、损失函数和优化器
model = SimpleModel()
loss_fn = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 创建虚拟数据用于演示
x = torch.randn(16, 10)
y = torch.randn(16, 1)

# 使用DeepSpeed初始化
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    args=None,  # 如果没有额外参数，设置为None
    lr_scheduler=None,
    config='ds_config.json'  # 需要创建一个DeepSpeed配置文件
)

# 假设我们有一个简单的训练循环
for epoch in range(10):
    outputs = model(x)
    loss = loss_fn(outputs, y)

    model.backward(loss)  # 替代loss.backward()
    model.step()  # 替代optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
