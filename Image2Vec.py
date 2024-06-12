import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import time

def normalize_vectors(vectors):
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    normalized_vectors = (vectors - mean) / std
    return normalized_vectors, mean, std

def denormalize_vectors(normalized_vectors, mean, std):
    original_vectors = normalized_vectors * std + mean
    return original_vectors

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_folder, vectors, transforms=None):
        self.image_folder = image_folder
        self.vectors = vectors
        self.transforms = transforms
        self.num_images = len(self.vectors)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image1_filename = f'LE11_normalized/strainfield-{idx}_LE11_cropped.png'
        image2_filename = f'LE22_normalized/strainfield-{idx}_LE22_cropped.png'
        image1_path = os.path.join(self.image_folder, image1_filename)
        image2_path = os.path.join(self.image_folder, image2_filename)
        image1 = Image.open(image1_path).convert('L')
        image2 = Image.open(image2_path).convert('L')

        if self.transforms:
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)


        vector = torch.tensor(self.vectors[idx], dtype=torch.float32)
        return image1, image2, vector

# 定义模型
# class ImagePairToVectorMLP(nn.Module):
#     def __init__(self):
#         super(ImagePairToVectorMLP, self).__init__()
#         input_dim = 2 * 81 * 81  # 对于两个300x300的图像
#         hidden_dim1 = 4096 # 第一个隐藏层的大小
#         hidden_dim2 = 1024   # 第二个隐藏层的大小
#         output_dim = 81     # 输出层的大小，根据您的目标设置
#
#         # 定义全连接层
#         self.fc1 = nn.Linear(input_dim, hidden_dim1)
#         self.dropout = nn.Dropout(0.1)
#         self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
#         self.fc3 = nn.Linear(hidden_dim2, output_dim)
#
#     def forward(self, x1, x2):
#         # 展平图像
#         x1 = x1.view(x1.size(0), -1)
#         x2 = x2.view(x2.size(0), -1)
#
#         # 合并两个图像的特征
#         x = torch.cat((x1, x2), dim=1)
#
#         # 通过全连接层
#         x = torch.tanh(self.fc1(x))
#         x = self.dropout(x)
#         x = torch.tanh(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#
#         return x

# class EnhancedImagePairToVectorNet(nn.Module):
#     def __init__(self):
#         super(EnhancedImagePairToVectorNet, self).__init__()
#         # Convolution layers
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Output: [32, 600, 600]
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Output: [64, 600, 600]
#         self.bn1 = nn.BatchNorm2d(32)
#         self.pool = nn.AvgPool2d(2, 2)  # Output: [64, 300, 300]
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: [128, 300, 300]
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: [256, 300, 300]
#         self.bn1 = nn.BatchNorm2d(128)
#         self.pool2 = nn.AvgPool2d(2, 2)  # Output: [128, 150, 150]
#         self.pool3 = nn.AvgPool2d(2, 2)  # Output: [128, 75, 75]
#         self.pool4 = nn.AvgPool2d(2, 2)  # Output: [128, 37, 37]
#
#         # Fully connected layers
#         fc_input_dim = 128 * 37 * 37 * 2  # Change based on the output of the last pooling layer
#         self.fc1 = nn.Linear(fc_input_dim, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 81)
#
#     def forward(self, x1, x2):
#         # Apply convolutional layers and pooling to the first image
#         x1 = self.pool(F.relu(self.conv2(F.relu(self.conv1(x1)))))
#         x1 = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x1)))))
#         x1 = self.pool3(x1)
#         x1 = self.pool4(x1)
#         x1 = x1.view(x1.size(0), -1)
#
#         # Apply convolutional layers and pooling to the second image
#         x2 = self.pool(F.relu(self.conv2(F.relu(self.conv1(x2)))))
#         x2 = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x2)))))
#         x2 = self.pool3(x2)
#         x2 = self.pool4(x2)
#         x2 = x2.view(x2.size(0), -1)
#
#         # Concatenate and pass through fully connected layers
#         x = torch.cat((x1, x2), dim=1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#
#         return x
class ImagePairToVectorNet(nn.Module):
    def __init__(self):
        super(ImagePairToVectorNet, self).__init__()
        # 调整卷积层参数
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2) # Output: [8, 150, 150]
        self.pool1 = nn.AvgPool2d(2, 2) # Output: [8, 75, 75]
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2) # Output: [16, 38, 38]
        self.pool2 = nn.AvgPool2d(2, 2) # Output: [16, 19, 19]
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # Output: [32, 19, 19]
        self.pool3 = nn.AvgPool2d(2, 2) # Output: [32, 9, 9]
        self.pool4 = nn.AvgPool2d(2, 2) # 添加额外的池化层以进一步减小尺寸 Output: [32, 4, 4]

        # 计算全连接层的输入尺寸
        fc_input_dim = 32 * 4 * 4 * 2 # 两个图像特征图合并后的总尺寸
        self.fc1 = nn.Linear(fc_input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 81)

    def forward(self, x1, x2):
        x1 = self.pool1(F.relu(self.conv1(x1)))
        x1 = self.pool2(F.relu(self.conv2(x1)))
        x1 = self.pool3(F.relu(self.conv3(x1)))
        x1 = self.pool4(x1) # 新增层
        x1 = x1.view(x1.size(0), -1)

        x2 = self.pool1(F.relu(self.conv1(x2)))
        x2 = self.pool2(F.relu(self.conv2(x2)))
        x2 = self.pool3(F.relu(self.conv3(x2)))
        x2 = self.pool4(x2) # 新增层
        x2 = x2.view(x2.size(0), -1)

        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 加载和预处理数据
df_vectors = pd.read_csv('/workspace/array_linear.csv', header=None).values
# 假设前1154个样本用于训练，后20个样本用于测试
train_vectors = df_vectors[:10000]
test_vectors = df_vectors[10000:10100]
train_vectors_normalized, mean_vector, std_vector = normalize_vectors(train_vectors)
test_vectors_normalized = (test_vectors - mean_vector) / std_vector

transform = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])


# 更新 CustomDataset 类以使用归一化向量
train_dataset = CustomDataset(image_folder='/workspace/', vectors=train_vectors_normalized, transforms=transform)
test_dataset = CustomDataset(image_folder='/workspace/', vectors=test_vectors_normalized, transforms=transform)

# 创建数据集和数据加载器

# train_dataset = CustomDataset(image_folder='/workspace/', vectors=train_vectors, transforms=transform)
# test_dataset = CustomDataset(image_folder='/workspace/', vectors=test_vectors, transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImagePairToVectorNet().to(device)
criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 正则化
# optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)

# 训练和验证过程
num_epochs = 100
batch_size = 8
n_train_batches = len(train_loader)
n_test_batches = len(test_loader)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_mae = 0
    train_preds, train_labels = [], []
    for image1_batch, image2_batch, vector_batch in train_loader:
        image1_batch = image1_batch.to(device)
        image2_batch = image2_batch.to(device)
        vector_batch = vector_batch.to(device)

        optimizer.zero_grad()
        outputs = model(image1_batch, image2_batch)
        loss = criterion(outputs, vector_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_mae += torch.mean(torch.abs(outputs - vector_batch)).item()

        # 反归一化训练集的输出
        outputs_denormalized = denormalize_vectors(outputs.detach().cpu().numpy(), mean_vector, std_vector)
        vector_batch_denormalized = denormalize_vectors(vector_batch.cpu().numpy(), mean_vector, std_vector)

        train_preds.extend(outputs_denormalized)
        train_labels.extend(vector_batch_denormalized)

    train_loss /= n_train_batches
    train_mae /= n_train_batches
    train_r2 = r2_score(np.concatenate(train_labels), np.concatenate(train_preds))

    # 验证过程
    model.eval()
    test_loss = 0
    test_mae = 0
    test_preds, test_labels = [], []
    for image1_batch, image2_batch, vector_batch in test_loader:
        image1_batch = image1_batch.to(device)
        image2_batch = image2_batch.to(device)
        vector_batch = vector_batch.to(device)

        outputs = model(image1_batch, image2_batch)
        loss = criterion(outputs, vector_batch)
        test_loss += loss.item()
        test_mae += torch.mean(torch.abs(outputs - vector_batch)).item()

        # 反归一化输出
        outputs_denormalized = denormalize_vectors(outputs.detach().cpu().numpy(), mean_vector, std_vector)
        vector_batch_denormalized = denormalize_vectors(vector_batch.cpu().numpy(), mean_vector, std_vector)

        test_preds.extend(outputs_denormalized)
        test_labels.extend(vector_batch_denormalized)

    test_loss /= n_test_batches
    test_mae /= n_test_batches
    test_r2 = r2_score(np.concatenate(test_labels), np.concatenate(test_preds))

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f},Train MAE: {train_mae:.4f}, Train R2: {train_r2:.4f}, Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}")

torch.save(model.state_dict(), 'Image2Vec.pth')
print("训练完成。")
