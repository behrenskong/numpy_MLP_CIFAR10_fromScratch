import numpy as np

class Activation:
    """激活函数基类"""
    def __init__(self):
        pass
        
    def forward(self, z):
        """前向传播"""
        pass
    
    def backward(self, z):
        """反向传播，计算梯度"""
        pass


class ReLU(Activation):
    """ReLU激活函数"""
    def __init__(self):
        super().__init__()
        
    def forward(self, z):
        """
        计算ReLU激活函数: f(z) = max(0, z)
        
        参数:
        z: 输入数据
        
        返回:
        ReLU激活后的结果
        """
        # 计算ReLU
        return np.maximum(0, z)
    
    def backward(self, z):
        """
        计算ReLU函数的梯度: f'(z) = 1 if z > 0 else 0
        
        参数:
        z: 输入值
        
        返回:
        ReLU梯度
        """
        # 正值导数为1，负值导数为0
        return (z > 0).astype(float)


class Sigmoid(Activation):
    """Sigmoid激活函数"""
    def __init__(self):
        super().__init__()
        
    def forward(self, z):
        """
        计算Sigmoid激活函数: f(z) = 1 / (1 + exp(-z))
        
        对负值使用等价公式exp(z)/(1+exp(z))以提高数值稳定性
        
        参数:
        z: 输入数据
        
        返回:
        Sigmoid激活后的结果
        """
        # 计算Sigmoid，同时优化数值稳定性
        output = np.zeros_like(z)
        
        # 对正负值分别处理以避免溢出问题
        neg_mask = z < 0
        pos_mask = ~neg_mask
        
        output[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
        exp_term = np.exp(z[neg_mask])
        output[neg_mask] = exp_term / (1.0 + exp_term)
        
        return output
    
    def backward(self, z, output=None):
        """
        计算Sigmoid函数的梯度: f'(z) = f(z) * (1 - f(z))
        
        参数:
        z: 输入值
        output: 如果提供，直接使用输出值计算梯度；否则重新计算
        
        返回:
        Sigmoid梯度
        """
        # 如果提供了output，直接用它计算梯度
        if output is not None:
            return output * (1 - output)
        
        # 否则重新计算sigmoid
        s = self.forward(z)
        return s * (1 - s)


class Softmax(Activation):
    """
    Softmax激活函数，将输入转换为概率分布
    
    softmax(z)_i = exp(z_i) / sum_j(exp(z_j))
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, z):
        """
        计算Softmax激活函数
        
        参数:
        z: 输入张量，形状为(batch_size, n_classes)
        
        返回:
        概率分布，形状与输入相同
        """
        # 为了数值稳定性，减去每行的最大值
        shifted_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted_z)
        output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        return output
    
    def backward(self, dL_dy=None, output=None):
        """
        计算Softmax函数关于输入的梯度
        
        Softmax梯度计算有两种情况:
        1. 与交叉熵损失结合：梯度简化为 y_pred - y_true，通过dL_dy参数传入
        2. 其他损失函数：需计算完整的Jacobian矩阵
        
        参数:
        dL_dy: 从损失函数反向传播的梯度
        output: Softmax的输出值
        
        返回:
        Softmax关于输入的梯度
        """
        # 与交叉熵损失结合的情况
        if dL_dy is not None:
            return dL_dy
        
        # 非交叉熵损失情况，需计算完整的Jacobian矩阵
        # 确保output已提供
        if output is None:
            raise ValueError("在计算Softmax梯度时必须提供output参数")
            
        batch_size = output.shape[0]
        n_classes = output.shape[1]
        jacobian = np.zeros((batch_size, n_classes, n_classes))
        
        # 对每个样本计算Jacobian矩阵
        for i in range(batch_size):
            for j in range(n_classes):
                for k in range(n_classes):
                    if j == k:
                        # 对角元素: ∂softmax_j/∂z_j = softmax_j * (1 - softmax_j)
                        jacobian[i, j, k] = output[i, j] * (1 - output[i, j])
                    else:
                        # 非对角元素: ∂softmax_j/∂z_k = -softmax_j * softmax_k
                        jacobian[i, j, k] = -output[i, j] * output[i, k]
        
        # 默认情况下，假设上游梯度为单位矩阵，用于测试
        identity_gradient = np.eye(n_classes)[np.newaxis, :, :] * np.ones((batch_size, 1, 1))
        result = np.zeros((batch_size, n_classes))
        
        # 对每个样本，计算梯度与Jacobian的乘积
        for i in range(batch_size):
            result[i] = np.sum(identity_gradient[i] * jacobian[i], axis=1)
            
        return result 