import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.preprocessing import StandardScaler
import time
from PIL import Image

# 处理图像的函数

def reassemble_image(vector, size=(60, 60)):
    """
    将展平的图像向量重组成图像
    :param vector: 展平的图像向量
    :param size: 重组图像的大小
    :return: 重组后的图像对象
    """
    image_array = vector.reshape(size)
    image = Image.fromarray(image_array)
    if image.mode != 'L':
        image = image.convert('L')
    return image

def save_images(vectors, folder_path, prefix='predicted_image', start_index=0):
    """
    将一系列图像向量保存为图像文件
    :param vectors: 包含展平图像向量的数组
    :param folder_path: 保存图像的目录路径
    :param prefix: 保存的图像文件名前缀
    :param start_index: 起始索引号
    """
    for i, vector in enumerate(vectors):
        image = reassemble_image(vector)
        image_file_path = os.path.join(folder_path, f'{prefix}_{i + start_index}.png')
        image.save(image_file_path)

def process_image_flat(path):
    img = Image.open(path).convert('L')  # 转换为灰度图
    img_array = np.array(img)

    # 确保图像大小是234x234
    if img_array.shape[0] != 234 or img_array.shape[1] != 234:
        raise ValueError("Image size is not 234x234 pixels")

    return img_array.flatten()  # 直接展平整个图像
def process_image(path):
    img = Image.open(path).convert('L')  # 转换为灰度图
    img = img.resize((60, 60))           # 调整大小
    return np.array(img).flatten()       # 展平为向量


def process_image_grid(path, grid_size=(26, 26)):
    img = Image.open(path).convert('L')  # 转换为灰度图
    img_array = np.array(img)

    # 确保图像大小是234x234
    if img_array.shape[0] != 234 or img_array.shape[1] != 234:
        raise ValueError("Image size is not 234x234 pixels")

    # 计算每个网格的大小
    grid_height, grid_width = img_array.shape[0] // grid_size[0], img_array.shape[1] // grid_size[1]

    # 初始化展平的网格像素值数组
    flattened_grid_pixels = []

    # 遍历每个网格
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # 获取网格的像素块
            grid = img_array[i*grid_height:(i+1)*grid_height, j*grid_width:(j+1)*grid_width]

            # 计算网格的平均像素值并添加到列表中
            avg_pixel_value = np.mean(grid)
            flattened_grid_pixels.append(avg_pixel_value)

    return np.array(flattened_grid_pixels)


# 构建两组图像路径
folder_path_1 = 'D:\\Dropbox\\2Dstrain\\LE11_normalized'
folder_path_2 = 'D:\\Dropbox\\2Dstrain\\LE22_normalized'
image_paths_1 = [os.path.join(folder_path_1, f'strainfield-{i}_LE11_cropped.png') for i in range(10100)]
image_paths_2 = [os.path.join(folder_path_2, f'strainfield-{i}_LE22_cropped.png') for i in range(10100)]

# 加载和处理两组图像，并将它们叠加
y = np.array([np.concatenate((process_image(path1), process_image(path2))) for path1, path2 in zip(image_paths_1, image_paths_2)])


# X = pd.read_csv('D:\\MLP\\feature_vectors.csv', header=None, nrows=10100)
# 其余数据处理部分
X = pd.read_csv('D:\\Dropbox\\2Dstrain\\array_linear.csv', header=None, nrows=10100)

train_size = 10000

X_train = X.values[:train_size]
X_test = X.values[train_size:]
Y_train = y[:train_size]
Y_test = y[train_size:]

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
Y_train_scaled = scaler_y.fit_transform(Y_train)
X_test_scaled = scaler_X.transform(X_test)
Y_test_scaled = scaler_y.transform(Y_test)
print(X_train_scaled)
print(Y_train_scaled)
# 120,300,580,780
model = MLPRegressor(hidden_layer_sizes=(200,1500),
                     activation='tanh',
                     solver='sgd',
                     alpha=0.0001,
                     batch_size='auto',
                     learning_rate='constant',
                     learning_rate_init=0.001,
                     max_iter=500000,
                     shuffle=True,
                     verbose=True,
                     random_state=1,
                     tol=0.0001)


starttime_inverse = time.time()
model.fit(X_train_scaled, Y_train_scaled)
endtime_inverse = time.time()
totaltime_inverse = endtime_inverse-starttime_inverse

score = model.score(X_test_scaled, Y_test_scaled)
Y_pred_mlp_inverse = model.predict(X_test_scaled)

Y_pred_mlp_inverse_original = scaler_y.inverse_transform(Y_pred_mlp_inverse)

#joblib.dump(model, "MLP_Image.m")
print(model.score(X_train_scaled, Y_train_scaled))
print(model.score(X_test_scaled, Y_test_scaled))
print(mean_squared_error(Y_test, Y_pred_mlp_inverse_original))
print(totaltime_inverse)
print("Model accuracy:", score)

predicted_vectors = Y_pred_mlp_inverse_original

# 定义保存图像的路径（请根据需要更改此路径）
output_folder = 'G:\\predictions'

# 保存前100个预测的图像
save_images(predicted_vectors[:100], output_folder)