import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from torch.autograd import Variable

def unnormalize(tensor, mean, std):
    """
    反归一化图像。
    :param tensor: 归一化的图像张量
    :param mean: 用于归一化的均值
    :param std: 用于归一化的标准偏差
    :return: 反归一化后的图像张量
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # 反向应用公式 (x - mean) / std
    return tensor
def gaussian(window_size, sigma):
    # First, create a tensor for the range values
    sequence = torch.arange(window_size).float() - window_size // 2
    # Then, apply the Gaussian formula using tensor operations
    gauss = torch.exp(-0.5 * (sequence / sigma) ** 2)
    # Finally, normalize the Gaussian window
    gauss = gauss / gauss.sum()
    return gauss


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=15, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# 定义数据集类
class VectorToImageDataset(Dataset):
    def __init__(self, vectors, image_folder, transforms=None, vector_min=None, vector_max=None):
        self.vectors = vectors
        self.image_folder = image_folder
        self.transforms = transforms
        self.vector_min = vectors.min(0) if vector_min is None else vector_min
        self.vector_max = vectors.max(0) if vector_max is None else vector_max
        print(self.vector_min)
        print(self.vector_max)

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        vector = torch.tensor(self.vectors[idx], dtype=torch.float32)
        # 归一化向量
        vector = (vector - self.vector_min) / (self.vector_max - self.vector_min)
        image_path = os.path.join(self.image_folder, f'strainfield-{idx}_LE11_cropped.png')
        image = Image.open(image_path).convert('L')

        if self.transforms:
            image = self.transforms(image)

        return vector, image

# 定义模型
# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""
#
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.double_conv(x)
# class VectorToImageUNet(nn.Module):
#     def __init__(self, input_dim, output_channels):
#         super(VectorToImageUNet, self).__init__()
#
#         # 将81维向量映射到一个具有空间维度的特征图（例如10x10）
#         self.fc = nn.Linear(input_dim, 256 * 10 * 10)
#
#         # U-Net结构
#         self.down1 = DoubleConv(256, 128)
#         self.down2 = DoubleConv(128, 64)
#         self.up1 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2)
#         self.conv1 = DoubleConv(128, 128)
#         self.up2 = nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2)
#         self.conv2 = DoubleConv(256, 256)
#         self.up3 = nn.ConvTranspose2d(256, 512, kernel_size=2, stride=2)
#         self.conv3 = DoubleConv(512, 512)
#         self.final_up = nn.ConvTranspose2d(512, output_channels, kernel_size=2, stride=2)
#
#     def forward(self, x):
#         x = self.fc(x)
#         x = F.relu(x)
#         x = x.view(-1, 256, 10, 10)  # 调整x的形状以匹配U-Net的输入尺寸
#
#         x = F.max_pool2d(self.down1(x), kernel_size=2, stride=2)
#         x = F.max_pool2d(self.down2(x), kernel_size=2, stride=2)
#         x = self.up1(x)
#         x = self.conv1(x)
#         x = self.up2(x)
#         x = self.conv2(x)
#         x = self.up3(x)
#         x = self.conv3(x)
#         x = self.final_up(x)
#
#         # 最终上采样到300x300
#         x = F.interpolate(x, size=(300, 300), mode='bilinear', align_corners=False)
#
#         return torch.sigmoid(x)


class VectorToImageNet(nn.Module):
    def __init__(self, vector_size):
        super(VectorToImageNet, self).__init__()
        # 设定初始图像大小为60x60
        initial_image_size = 60

        self.fc1 = nn.Linear(vector_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, initial_image_size * initial_image_size)

        # 上采样层
        self.upsample1 = nn.ConvTranspose2d(1, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample4 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 确保输出在0到1之间
        x = x.view(-1, 1, 60, 60)  # 调整形状以匹配60x60图像尺寸

        # 上采样过程
        x = F.relu(self.upsample1(x))
        x = F.relu(self.upsample2(x))
        x = F.relu(self.upsample3(x))
        x = torch.sigmoid(self.upsample4(x))  # 最终图像为1个通道（灰度图）

        return x

# class VectorToImageNet(nn.Module):
#     def __init__(self, vector_size, image_size):
#         super(VectorToImageNet, self).__init__()
#         self.fc1 = nn.Linear(vector_size, 512)
#         self.fc2 = nn.Linear(512, 2024)
#         self.fc3 = nn.Linear(2024, 10000)
#         self.fc4 = nn.Linear(10000, image_size * image_size)  # 假设图像大小为image_size x image_size
#         self.image_size = image_size
#
#     def forward(self, x):
#         x = torch.tanh(self.fc1(x))
#         x = torch.tanh(self.fc2(x))
#         x = torch.tanh(self.fc3(x))
#         x = torch.sigmoid(self.fc4(x))  # 使用sigmoid来确保输出在0到1之间
#         x = x.view(-1, 1, self.image_size, self.image_size)  # 调整形状以匹配图像尺寸
#         return x


def calculate_metrics(y_true, y_pred):
    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
    return mean_squared_error(y_true_flat, y_pred_flat), r2_score(y_true_flat, y_pred_flat)


def calculate_ssim(y_true, y_pred):
    ssim_scores = []
    for true_img, pred_img in zip(y_true, y_pred):
        true_img = true_img.squeeze()  # 移除通道维度
        pred_img = pred_img.squeeze()  # 移除通道维度
        score = ssim(true_img, pred_img, data_range=pred_img.max() - pred_img.min())
        ssim_scores.append(score)
    return np.mean(ssim_scores)


# 加载和预处理数据
df_vectors = pd.read_csv('/workspace/array_linear.csv', header=None).values
train_vectors = df_vectors[:10000]
test_vectors = df_vectors[10000:10100]

# 创建数据集和数据加载器
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    # 假设图像是灰度的，均值和标准差设置为0.5 (实际值可能会有所不同)
    # transforms.Normalize((0.5,), (0.5,))
])

vector_min = train_vectors.min(0)
vector_max = train_vectors.max(0)

train_dataset = VectorToImageDataset(train_vectors, '/workspace/LE11_normalized', transform, vector_min, vector_max)
test_dataset = VectorToImageDataset(test_vectors, '/workspace/LE11_normalized', transform, vector_min, vector_max)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vector_size = len(train_vectors[0])
image_size = 300  # 假设图像大小为300x300
# model = VectorToImageUNet(input_dim=81, output_channels=1).to(device)
model = VectorToImageNet(vector_size).to(device)
criterion = nn.L1Loss()
ssim_loss = SSIM(window_size=15, size_average=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.8)

# 训练和验证过程
num_epochs = 600
print("start training")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    all_outputs = []
    all_images = []

    # 训练过程
    for vector, image in train_loader:
        vector = vector.to(device).float()
        image = image.to(device)
        optimizer.zero_grad()
        output = model(vector)
        loss = criterion(output, image)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        all_outputs.append(output.detach().cpu())
        all_images.append(image.cpu())

    train_loss /= len(train_loader)
    train_ssim = calculate_ssim(np.concatenate([img.numpy() for img in all_images]),
                                np.concatenate([img.numpy() for img in all_outputs]))
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train SSIM: {train_ssim:.4f}")

    # 测试过程
    model.eval()
    test_loss = 0
    all_test_outputs = []
    all_test_images = []
    with torch.no_grad():
        for vector, image in test_loader:
            vector = vector.to(device).float()
            image = image.to(device)
            output = model(vector)
            test_loss += criterion(output, image).item()
            all_test_outputs.append(output.cpu())
            all_test_images.append(image.cpu())

    test_loss /= len(test_loader)
    test_ssim = calculate_ssim(np.concatenate([img.numpy() for img in all_test_images]),
                               np.concatenate([img.numpy() for img in all_test_outputs]))
    print(f"Test Loss: {test_loss:.4f}, Test SSIM: {test_ssim:.4f}")

    # 可视化：反归一化并绘制测试集中第一个向量对应的LE11图像
    first_vector, original_image = test_dataset[0]
    first_vector = first_vector.to(device).float()
    first_image_output = model(first_vector.unsqueeze(0)).detach().cpu().squeeze(0)

    # 反归一化图像
    # first_image_output_unnorm = unnormalize(first_image_output.clone(), mean, std).squeeze(0)

    original_image_np = original_image.squeeze().numpy()
    first_image_output_np = first_image_output.squeeze().numpy()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(first_image_output_np, cmap='gray')
    plt.title(f"Generated LE11 Image at Epoch {epoch + 1}")
    plt.subplot(1, 2, 2)
    plt.imshow(original_image_np, cmap='gray')
    plt.title("Original LE11 Image")
    plt.show()

torch.save(model.state_dict(), 'vector_to_image_model.pth')
print("Training completed.")
