import numpy as np

class Optimizer:
    """优化器基类"""
    def __init__(self):
        pass
    
    def update(self, model):
        """更新模型参数"""
        pass


class SGD(Optimizer):
    """
    随机梯度下降(SGD)
    
    支持momentum，学习率通过ExponentialScheduler进行控制
    """
    def __init__(self, learning_rate=0.01, momentum=0.0):
        """
        初始化SGD优化器
        
        参数:
        learning_rate: 学习率
        momentum: 动量系数
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate  # 保存初始学习率
        self.momentum = momentum
        self.iterations = 0
        self.velocity = {}  # 储存动量项
    
    def update(self, model):
        """
        根据梯度更新模型参数
        
        参数:
        model: 神经网络模型
        """
        # 如果还没有初始化动量项
        if not self.velocity:
            for layer_name, layer in model.layers.items():
                self.velocity[layer_name] = {
                    'W': np.zeros_like(layer.params['W']),
                    'b': np.zeros_like(layer.params['b'])
                }
        
        # 使用动量SGD更新参数
        for layer_name, layer in model.layers.items():
            for param_name in ['W', 'b']:
                # 计算动量项
                self.velocity[layer_name][param_name] = (
                    self.momentum * self.velocity[layer_name][param_name] - 
                    self.learning_rate * layer.grads[param_name]
                )
                # 更新参数
                layer.params[param_name] += self.velocity[layer_name][param_name]
        
        self.iterations += 1


class ExponentialScheduler:
    """
    指数衰减学习率调度器
    
    每stage_length步，学习率乘以衰减系数decay。
    这是推荐用于SGD的标准学习率调度方法。
    """
    def __init__(self, initial_lr=0.01, stage_length=500, decay=0.95, staircase=True):
        """
        初始化指数衰减学习率调度器
        
        参数:
        initial_lr: 初始学习率
        stage_length: 每个阶段的长度（步数）。通常设置为1个epoch的步数，以便每个epoch衰减一次。
        decay: 衰减率。例如，0.95表示每个阶段后学习率衰减为原来的95%。
        staircase: 是否使用阶梯式衰减（True）或连续衰减（False）
        """
        self.initial_lr = initial_lr
        self.stage_length = stage_length
        self.decay = decay
        self.staircase = staircase
        
    def __call__(self, step):
        """
        计算当前步的学习率
        
        参数:
        step: 当前训练步数
        
        返回:
        当前的学习率
        """
        cur_stage = step / self.stage_length
        if self.staircase:
            # 阶梯式衰减，每stage_length步更新一次
            cur_stage = np.floor(cur_stage)
        return self.initial_lr * (self.decay ** cur_stage) 