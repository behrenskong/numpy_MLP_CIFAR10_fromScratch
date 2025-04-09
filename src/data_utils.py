import numpy as np
import pickle
import os
import urllib.request
import tarfile

def to_one_hot(y, num_classes=10):
    """
    将标签转换为one-hot编码
    
    参数:
    y: 类别标签
    num_classes: 类别数量
    
    返回:
    one-hot编码标签
    """
    return np.eye(num_classes)[y]

def download_cifar10(data_dir='./data'):
    """
    下载CIFAR-10数据集
    
    参数:
    data_dir: 数据存储目录
    
    返回:
    CIFAR-10数据所在目录路径
    """
    cifar10_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    
    # 如果数据已存在，直接返回
    if os.path.exists(cifar10_dir):
        print(f"CIFAR-10数据集已存在于 {cifar10_dir}")
        return cifar10_dir
    
    # 确保数据目录存在
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # CIFAR-10数据集URL
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = os.path.join(data_dir, 'cifar-10-python.tar.gz')
    
    # 下载数据集
    print(f"正在下载CIFAR-10数据集...")
    urllib.request.urlretrieve(url, filename)
    print(f"下载完成。正在解压...")
    
    # 解压数据集
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    
    print(f"解压完成，数据集保存在 {cifar10_dir}")
    
    # 删除压缩包
    os.remove(filename)
    
    return cifar10_dir

def load_batch(batch_file):
    """
    加载CIFAR-10批次文件
    
    参数:
    batch_file: 批次文件路径
    
    返回:
    数据和标签
    """
    with open(batch_file, 'rb') as f:
        batch_data = pickle.load(f, encoding='bytes')
    
    # 转换为numpy数组
    data = batch_data[b'data']
    labels = np.array(batch_data[b'labels'])
    
    return data, labels

def load_cifar10_data(data_dir='./data', normalize=True, flatten=True, one_hot=True, validation_size=0.1, seed=42):
    """
    加载CIFAR-10数据集
    
    参数:
    data_dir: 数据存储目录
    normalize: 是否将像素值归一化到[0,1]区间
    flatten: 是否将图像展平为一维向量
    one_hot: 是否将标签转换为one-hot编码
    validation_size: 验证集比例
    seed: 随机数种子，确保结果可重现
    
    返回:
    (x_train, y_train), (x_val, y_val), (x_test, y_test)元组
    """
    # 设置随机种子确保可重现
    np.random.seed(seed)
    
    # 下载数据集（如果需要）
    cifar10_dir = download_cifar10(data_dir)
    
    # 加载训练数据
    x_train = []
    y_train = []
    
    for i in range(1, 6):
        batch_file = os.path.join(cifar10_dir, f'data_batch_{i}')
        data, labels = load_batch(batch_file)
        x_train.append(data)
        y_train.append(labels)
    
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    
    # 加载测试数据
    test_batch_file = os.path.join(cifar10_dir, 'test_batch')
    x_test, y_test = load_batch(test_batch_file)
    
    # 随机划分验证集
    if validation_size > 0:
        # 随机打乱索引
        indices = np.random.permutation(len(x_train))
        n_val = int(len(x_train) * validation_size)
        
        # 使用随机索引划分训练集和验证集
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        x_val = x_train[val_indices]
        y_val = y_train[val_indices]
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]
        
        # 检查类别分布是否均衡
        if not one_hot:
            train_class_dist = np.bincount(y_train) / len(y_train)
            val_class_dist = np.bincount(y_val) / len(y_val)
            print("训练集类别分布:", train_class_dist)
            print("验证集类别分布:", val_class_dist)
    else:
        x_val = np.array([])
        y_val = np.array([])
    
    # 重塑图像: (N, 3072) -> (N, 32, 32, 3)
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    if len(x_val) > 0:
        x_val = x_val.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # 展平图像为一维向量
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        if len(x_val) > 0:
            x_val = x_val.reshape(x_val.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    
    # 归一化像素值到[0,1]
    if normalize:
        x_train = x_train.astype('float32') / 255.0
        if len(x_val) > 0:
            x_val = x_val.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    
    # 标准化数据（减均值除以标准差）
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0) + 1e-8
    x_train = (x_train - mean) / std
    if len(x_val) > 0:
        x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std
    
    # 将标签转换为one-hot编码
    if one_hot:
        y_train = to_one_hot(y_train)
        if len(y_val) > 0:
            y_val = to_one_hot(y_val)
        y_test = to_one_hot(y_test)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

class BatchIterator:
    """批处理迭代器"""
    
    def __init__(self, x, y, batch_size=128, shuffle=True, seed=42):
        """
        初始化批处理迭代器
        
        参数:
        x: 输入数据
        y: 标签数据
        batch_size: 批大小
        shuffle: 是否打乱数据
        seed: 随机数种子，确保结果可重现
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.RandomState(seed)  # 创建独立的随机数生成器
        self.num_samples = x.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / batch_size))
        self.reset()
    
    def reset(self):
        """重置迭代器并在需要时打乱数据"""
        self.current_batch = 0
        if self.shuffle:
            indices = self.rng.permutation(self.num_samples)
            self.x = self.x[indices]
            self.y = self.y[indices]
    
    def __iter__(self):
        """迭代器协议"""
        return self
    
    def __next__(self):
        """获取下一批数据"""
        if self.current_batch >= self.num_batches:
            self.reset()
            raise StopIteration
        
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        
        batch_x = self.x[start_idx:end_idx]
        batch_y = self.y[start_idx:end_idx]
        
        self.current_batch += 1
        
        return batch_x, batch_y 