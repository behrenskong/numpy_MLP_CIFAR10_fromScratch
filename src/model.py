import numpy as np
import pickle
from src.layers import FullyConnectedLayer, DropoutLayer
from src.activation import ReLU, Sigmoid, Softmax
from src.loss import CrossEntropyLoss
from contextlib import contextmanager
import threading

# 全局上下文标志
_no_grad_context = threading.local()
_no_grad_context.enabled = False

class ThreeLayerNet:
    """
    三层神经网络分类器
    
    结构：输入层 -> 隐藏层1 -> Dropout -> 隐藏层2 -> Dropout -> 输出层(Softmax)
    """
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, 
                 activation='relu', weight_decay=0.0, dropout=0.0):
        """
        初始化三层神经网络
        
        参数:
        input_size: 输入特征维度
        hidden1_size: 第一隐藏层大小
        hidden2_size: 第二隐藏层大小
        output_size: 输出类别数
        activation: 激活函数类型 ('relu' 或 'sigmoid')
        weight_decay: L2正则化系数
        dropout: Dropout正则化比率（设为0则不使用Dropout）
        """
        self.weight_decay = weight_decay
        self.use_dropout = dropout > 0
        self.is_training = True  # 训练模式标志
        
        # 初始化网络层
        self.layers = {}
        self.layers['fc1'] = FullyConnectedLayer(input_size, hidden1_size)
        self.layers['fc2'] = FullyConnectedLayer(hidden1_size, hidden2_size)
        self.layers['fc3'] = FullyConnectedLayer(hidden2_size, output_size)
        
        # 初始化Dropout层（如果启用）
        if self.use_dropout:
            self.dropout1 = DropoutLayer(dropout)
            self.dropout2 = DropoutLayer(dropout)
        
        # 初始化激活函数
        if activation.lower() == 'relu':
            self.activation1 = ReLU()
            self.activation2 = ReLU()
        elif activation.lower() == 'sigmoid':
            self.activation1 = Sigmoid()
            self.activation2 = Sigmoid()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 输出层使用Softmax激活
        self.softmax = Softmax()
        
        # 损失函数
        self.loss_func = CrossEntropyLoss()
        
        # 存储中间激活值，用于反向传播
        self.activation_cache = {}
        
    def forward(self, x):
        """
        前向传播
        
        参数:
        x: 形状为(batch_size, input_size)的输入数据
        
        返回:
        输出层的预测概率
        """
        # 缓存输入数据
        self.activation_cache['x'] = x
        
        # 第一层: 输入 -> 隐藏层1
        z1 = self.layers['fc1'].forward(x)
        a1 = self.activation1.forward(z1)
        
        # 只在需要梯度时缓存中间结果
        if not _no_grad_context.enabled:
            self.activation_cache['z1'] = z1
            self.activation_cache['a1'] = a1
        
        # Dropout层1（如果启用）
        if self.use_dropout and self.is_training:
            a1 = self.dropout1.forward(a1)
            if not _no_grad_context.enabled:
                self.activation_cache['a1_dropout'] = a1
        
        # 第二层: 隐藏层1 -> 隐藏层2
        z2 = self.layers['fc2'].forward(a1)
        a2 = self.activation2.forward(z2)
        
        # 只在需要梯度时缓存中间结果
        if not _no_grad_context.enabled:
            self.activation_cache['z2'] = z2
            self.activation_cache['a2'] = a2
        
        # Dropout层2（如果启用）
        if self.use_dropout and self.is_training:
            a2 = self.dropout2.forward(a2)
            if not _no_grad_context.enabled:
                self.activation_cache['a2_dropout'] = a2
        
        # 第三层: 隐藏层2 -> 输出层
        z3 = self.layers['fc3'].forward(a2)
        
        # 只在需要梯度时缓存中间结果
        if not _no_grad_context.enabled:
            self.activation_cache['z3'] = z3
        
        # Softmax输出
        y_pred = self.softmax.forward(z3)
        
        # 只在需要梯度时缓存预测结果
        if not _no_grad_context.enabled:
            self.activation_cache['y_pred'] = y_pred
        
        return y_pred
    
    def loss(self, y, y_pred=None):
        """
        计算损失值
        
        参数:
        y: 真实标签 (one-hot编码)
        y_pred: 模型预测值（如果为None，使用上次前向传播的结果）
        
        返回:
        loss: 总损失值 (交叉熵损失 + L2正则化损失)
        """
        # 如果没有提供预测值，使用缓存中的预测结果
        if y_pred is None:
            y_pred = self.activation_cache['y_pred']
        
        # 计算交叉熵损失
        data_loss = self.loss_func.forward(y, y_pred)
        
        # 计算L2正则化损失
        reg_loss = 0
        if self.weight_decay > 0:
            for layer_name in self.layers:
                W = self.layers[layer_name].params['W']
                reg_loss += 0.5 * self.weight_decay * np.sum(W * W)
        
        return data_loss + reg_loss
    
    def backward(self, y, y_pred=None):
        """
        反向传播计算梯度
        
        参数:
        y: 真实标签 (one-hot编码)
        y_pred: 模型预测值（如果为None，使用上次前向传播的结果）
        """
        # 如果禁用梯度计算，直接返回
        if _no_grad_context.enabled:
            return None
        
        # 如果没有提供预测值，使用缓存中的预测结果
        if y_pred is None:
            y_pred = self.activation_cache['y_pred']
        
        batch_size = y.shape[0]
        
        # 计算输出层的梯度 (softmax + 交叉熵的组合梯度)
        # dL/dz3 = y_pred - y_true
        dz3 = self.loss_func.backward(y, y_pred)
        
        # 反向传播：输出层 -> 隐藏层2
        # dL/da2 = dL/dz3 * dz3/da2 = dL/dz3 * W3^T
        da2 = self.layers['fc3'].backward(dz3)
        
        # 反向传播：Dropout层2（如果启用）
        if self.use_dropout and self.is_training:
            da2 = self.dropout2.backward(da2)
        
        # 反向传播：隐藏层2激活函数
        dz2 = da2 * self.activation2.backward(self.activation_cache['z2'])
        
        # 反向传播：隐藏层2 -> 隐藏层1
        # dL/da1 = dL/dz2 * dz2/da1 = dL/dz2 * W2^T
        da1 = self.layers['fc2'].backward(dz2)
        
        # 反向传播：Dropout层1（如果启用）
        if self.use_dropout and self.is_training:
            da1 = self.dropout1.backward(da1)
        
        # 反向传播：隐藏层1激活函数
        dz1 = da1 * self.activation1.backward(self.activation_cache['z1'])
        
        # 反向传播：隐藏层1 -> 输入层
        # dL/dx = dL/dz1 * dz1/dx = dL/dz1 * W1^T
        dx = self.layers['fc1'].backward(dz1)
        
        # 添加L2正则化梯度
        if self.weight_decay > 0:
            for layer_name in self.layers:
                self.layers[layer_name].grads['W'] += self.weight_decay * self.layers[layer_name].params['W']
    
    def train(self, mode=True):
        """
        设置模型训练/评估模式
        
        参数:
        mode: 如果为True，设置为训练模式；否则设置为评估模式
        """
        self.is_training = mode
        if self.use_dropout:
            if mode:
                self.dropout1.train()
                self.dropout2.train()
            else:
                self.dropout1.eval()
                self.dropout2.eval()
        return self
    
    def eval(self):
        """
        设置模型为评估模式
        """
        return self.train(False)
    
    def zero_grad(self):
        """清零所有参数的梯度"""
        for layer_name in self.layers:
            self.layers[layer_name].zero_grad()
    
    def get_params(self):
        """
        获取模型参数
        
        返回:
        所有层的参数
        """
        params = {}
        for layer_name, layer in self.layers.items():
            params[layer_name] = {'W': layer.params['W'].copy(), 
                                 'b': layer.params['b'].copy()}
        return params
    
    def set_params(self, params):
        """
        设置模型参数
        
        参数:
        params: 模型参数
        """
        for layer_name, layer_params in params.items():
            self.layers[layer_name].params['W'] = layer_params['W'].copy()
            self.layers[layer_name].params['b'] = layer_params['b'].copy()
    
    def save_model(self, filepath):
        """
        保存模型参数到文件
        
        参数:
        filepath: 保存路径
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.get_params(), f)
        print(f"模型已保存到 {filepath}")
    
    def load_model(self, filepath):
        """
        从文件加载模型参数
        
        参数:
        filepath: 模型文件路径
        """
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.set_params(params)
        print(f"模型参数已从 {filepath} 加载...")
    
    @contextmanager
    def no_grad(self):
        """上下文管理器，禁用梯度计算"""
        # 保存当前状态
        old_enabled = _no_grad_context.enabled
        # 设置为禁用梯度计算
        _no_grad_context.enabled = True
        try:
            yield
        finally:
            # 恢复原来的状态
            _no_grad_context.enabled = old_enabled
    
    def predict(self, x):
        """
        预测类别
        
        参数:
        x: 输入数据
        
        返回:
        预测的类别索引
        """
        # 设置为评估模式
        with self.no_grad():
            # 前向传播
            y_pred = self.forward(x)
            # 返回预测类别
            return np.argmax(y_pred, axis=1)