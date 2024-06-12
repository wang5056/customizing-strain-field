import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import save_image
import os
import time
# import pytorch_ssim

def gaussian_window(size, sigma):
    x = torch.arange(0, size, 1, dtype=torch.float32)
    gauss = torch.exp(-torch.pow(x - size // 2, 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

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
class CustomDataset(Dataset):
    def __init__(self, base_dir, transform_lr=None, transform_hr=None, range_start=0, range_end=None):
        self.base_dir = base_dir
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

        # 根据 range 生成文件名列表
        self.images = [f"strainfield-{i}_LE22_cropped.png" for i in range(range_start, range_end)]

    def __getitem__(self, index):
        lr_image_path = os.path.join(self.base_dir, "LE22_resized", self.images[index])
        hr_image_path = os.path.join(self.base_dir, "LE22_cropped_0.35-0.75", self.images[index])

        try:
            lr_image = Image.open(lr_image_path).convert('L')
            hr_image = Image.open(hr_image_path).convert('L')
        except IOError as e:
            print(f"Error opening image: {e}")
            print(f"Image path: {hr_image_path}")
            # 返回占位符图像
            lr_placeholder = torch.zeros([1, 60, 60])
            hr_placeholder = torch.zeros([1, 240, 240])
            return lr_placeholder, hr_placeholder

        # 应用转换
        if self.transform_lr:
            lr_image = self.transform_lr(lr_image)
        if self.transform_hr:
            hr_image = self.transform_hr(hr_image)

        return lr_image, hr_image

    def __len__(self):
        return len(self.images)

# 转换器
transform_lr = transforms.Compose([
    transforms.Resize((60, 60)),
    transforms.ToTensor()
])

transform_hr = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor()
])

# 文件夹路径
base_dir = '/workspace/'
image_dir = 'high_res_pre/'
# 创建数据集
train_dataset = CustomDataset(base_dir=base_dir, transform_lr=transform_lr, transform_hr=transform_hr, range_end=10000)
test_dataset = CustomDataset(base_dir=base_dir, transform_lr=transform_lr, transform_hr=transform_hr, range_start=10000, range_end=10100)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

# 超参数
batch_size = 8
num_epochs = 400


# SRGAN模型构建
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * upscale_factor ** 2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)  # 1x1 卷积调整通道数

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        x = self.conv1x1(x)  # 应用 1x1 卷积
        return x

class Generator(nn.Module):
    def __init__(self, n_residual_blocks=16):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(n_residual_blocks)]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.upsample_blocks = nn.Sequential(
            UpsampleBLock(64, 64, 2),  # 60x60 -> 120x120
            UpsampleBLock(64, 64, 2),  # 60x60 -> 120x120
            # UpsampleBLock(64, 64, 2),  # 60x60 -> 120x120
            # UpsampleBLock(64, 64, 3),  # 120x120 -> 360x360
            # UpsampleBLock(64, 64, 5)   # 360x360 -> 1800x1800
        )

        self.conv3 = nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = self.residual_blocks(x)
        x = self.conv2(x)
        x = x + residual
        x = self.upsample_blocks(x)
        x = self.conv3(x)
        return x

class Discriminator(nn.Module):
    """
    SRGAN的判别器
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(1, 64, first_block=True),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()  # 添加 Sigmoid 激活函数
        )

    def forward(self, img):
        return self.model(img)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 损失函数和优化器
criterion_gan = nn.BCELoss().to(device)
criterion_content = nn.MSELoss().to(device) # 添加内容损失
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.00025)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
# optimizer_g = torch.optim.SGD(generator.parameters(), lr=0.0025, momentum=0.9)
# optimizer_d = torch.optim.SGD(discriminator.parameters(), lr=0.0001, momentum=0.9)
best_ssim = 0.0
best_epoch = 0
# 训练过程
for epoch in range(num_epochs):
    start_time = time.time()
    for lr_imgs, hr_imgs in train_loader:
        # 将数据移至 GPU
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        real_labels = torch.ones(batch_size).to(device)*0.9
        fake_labels = torch.zeros(batch_size).to(device)

        # 训练判别器
        optimizer_d.zero_grad()
        real_outputs = discriminator(hr_imgs)
        real_outputs = real_outputs.squeeze()  # 调整为 (batch_size,)
        d_loss_real = criterion_gan(real_outputs, real_labels.view_as(real_outputs))  # 调整标签形状以匹配输出
        fake_images = generator(lr_imgs)
        fake_outputs = discriminator(fake_images.detach())
        fake_outputs = fake_outputs.squeeze()  # 调整为 (batch_size,)
        d_loss_fake = criterion_gan(fake_outputs, fake_labels.view_as(fake_outputs))  # 同样调整标签形状
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # 训练生成器
        optimizer_g.zero_grad()
        fake_images = generator(lr_imgs)
        outputs = discriminator(fake_images)
        outputs = outputs.squeeze()  # 调整输出形状
        g_loss_gan = criterion_gan(outputs, real_labels)
        g_loss_content = criterion_content(fake_images, hr_imgs)
        g_loss = g_loss_gan + g_loss_content  # 合并GAN和内容损失
        g_loss.backward()
        optimizer_g.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss D: {d_loss.item()}, Loss G: {g_loss.item()}')


# 测试模型

    generator.eval()
    total_ssim = 0.0
    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            sr_imgs = generator(lr_imgs)
            for i in range(lr_imgs.size(0)):
                total_ssim += ssim(sr_imgs[i].unsqueeze(0), hr_imgs[i].unsqueeze(0))

    average_ssim = total_ssim / len(test_loader.dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average SSIM: {average_ssim:.4f}')
    end_time = time.time()
    total_time = end_time-start_time
    print(f'Epoch Time: {total_time/60}')
    # 检查并保存最佳模型
    if average_ssim > best_ssim:
        best_ssim = average_ssim
        best_epoch = epoch
        torch.save(generator.state_dict(), os.path.join(base_dir, 'LE22_60-240.pth'))
        print(best_ssim)

# 加载并保存最佳模型生成的图像
    generator.load_state_dict(torch.load(os.path.join(base_dir, 'LE22_60-240.pth')))
    generator.eval()
    with torch.no_grad():
        for idx, (lr_imgs, _) in enumerate(test_loader):
            lr_imgs = lr_imgs.to(device)
            sr_imgs = generator(lr_imgs)
            for i, img in enumerate(sr_imgs):
                save_image(img, os.path.join(base_dir, image_dir,
                                             f'high_res_image_LE22_epoch_{idx * test_loader.batch_size + i}.png'))
