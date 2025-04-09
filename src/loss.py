import numpy as np

class Loss:
    """损失函数基类"""
    def __init__(self):
        pass
        
    def forward(self, y_true, y_pred):
        """计算损失值"""
        pass
    
    def backward(self, y_true, y_pred):
        """计算损失函数关于预测值的梯度"""
        pass


class CrossEntropyLoss(Loss):
    """
    交叉熵损失函数，常用于多分类问题
    
    交叉熵定义为：L = -sum_i(y_true_i * log(y_pred_i))
    
    在神经网络中，通常与Softmax激活函数组合使用，梯度简化为y_pred - y_true
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, y_true, y_pred):
        """
        计算交叉熵损失
        
        参数:
        y_true: 形状为(batch_size, n_classes)的one-hot编码标签
        y_pred: 形状为(batch_size, n_classes)的模型预测概率
        
        返回:
        交叉熵损失值
        """
        # 防止数值问题（避免log(0)）
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # 计算交叉熵损失
        batch_size = y_true.shape[0]
        
        # 标准方式计算交叉熵：每个样本的每个类别都计算损失
        loss = -np.sum(y_true * np.log(y_pred)) / batch_size
        
        return loss
    
    def backward(self, y_true, y_pred):
        """
        计算交叉熵损失关于预测值的梯度
        
        参数:
        y_true: 形状为(batch_size, n_classes)的one-hot编码标签
        y_pred: 形状为(batch_size, n_classes)的模型预测概率
        
        返回:
        损失函数关于预测值的梯度 (y_pred - y_true)/batch_size
        """
        # 交叉熵损失+softmax的组合梯度为(y_pred - y_true)/batch_size
        batch_size = y_true.shape[0]
        return (y_pred - y_true) / batch_size 