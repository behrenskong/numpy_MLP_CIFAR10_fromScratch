import numpy as np

class Layer:
    """神经网络层基类"""
    def __init__(self):
        self.params = {}
        self.grads = {}
    
    def forward(self, inputs):
        """前向传播"""
        pass
    
    def backward(self, dout):
        """反向传播"""
        pass


class FullyConnectedLayer(Layer):
    """全连接层实现"""
    def __init__(self, input_size, output_size):
        """
        初始化全连接层
        
        参数:
        input_size: 输入特征维度
        output_size: 输出特征维度
        """
        super().__init__()
        
        # He初始化 - 适合ReLU激活函数
        # np.sqrt(2.0 / input_size)为标准差，而非方差
        scale = np.sqrt(2.0 / input_size)
        self.params['W'] = np.random.randn(input_size, output_size) * scale
        self.params['b'] = np.zeros(output_size)
        
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])
        
        # 缓存输入，用于反向传播
        self.inputs = None
    
    def forward(self, inputs):
        """
        前向传播计算: outputs = inputs @ W + b
        
        参数:
        inputs: 形状为(batch_size, input_size)的输入数据
        
        返回:
        形状为(batch_size, output_size)的输出
        """
        self.inputs = inputs  # 缓存输入用于反向传播
        outputs = np.dot(inputs, self.params['W']) + self.params['b']
        return outputs
    
    def backward(self, dout):
        """
        反向传播计算
        
        参数:
        dout: 形状为(batch_size, output_size)的输出梯度
        
        返回:
        形状为(batch_size, input_size)的输入梯度
        """
        # 计算权重梯度: dW = inputs^T @ dout
        self.grads['W'] = np.dot(self.inputs.T, dout)
        
        # 计算偏置梯度: db = sum(dout, axis=0)
        self.grads['b'] = np.sum(dout, axis=0)
        
        # 计算上一层的梯度: dinputs = dout @ W^T
        dinputs = np.dot(dout, self.params['W'].T)
        
        return dinputs
    
    def zero_grad(self):
        """清零梯度"""
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])


class DropoutLayer(Layer):
    """
    Dropout正则化层
    
    在训练过程中随机将一部分神经元的输出置为0，防止过拟合
    """
    def __init__(self, dropout_rate=0.5):
        """
        初始化Dropout层
        
        参数:
        dropout_rate: 丢弃率，表示有多大比例的神经元输出会被置为0
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None
        self.is_training = True  # 控制是否处于训练模式
        
    def forward(self, inputs):
        """
        前向传播：在训练时随机丢弃一些神经元，测试时对所有神经元进行缩放
        
        参数:
        inputs: 形状为(batch_size, n_features)的输入数据
        
        返回:
        应用dropout后的输出
        """
        # 训练模式：随机将一部分输入置为0
        if self.is_training:
            # 创建一个随机mask，保留概率为(1-dropout_rate)
            self.mask = np.random.rand(*inputs.shape) > self.dropout_rate
            
            # 对保留的值进行缩放，使期望值保持不变（inverted dropout技术）
            scale = 1 / (1 - self.dropout_rate)
            outputs = inputs * self.mask * scale
        # 测试模式：不使用dropout，保持输入不变
        else:
            outputs = inputs
            
        return outputs
    
    def backward(self, dout):
        """
        反向传播：传递梯度，但对被丢弃的神经元不传递梯度
        
        参数:
        dout: 上游梯度
        
        返回:
        对输入的梯度
        """

        # 反向传播时应用相同的mask和缩放
        scale = 1 / (1 - self.dropout_rate)
        dinputs = dout * self.mask * scale

        return dinputs
    
    def train(self):
        """设置为训练模式"""
        self.is_training = True
        
    def eval(self):
        """设置为评估模式"""
        self.is_training = False 