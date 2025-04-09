import numpy as np
import os
import pickle
from src.model import ThreeLayerNet
from src.data_utils import load_cifar10_data, BatchIterator

def load_and_test_model(model_path, test_data, batch_size=128):
    """
    加载模型并在测试集上评估性能
    
    参数:
    model_path: 模型文件路径
    test_data: 测试数据元组 (x_test, y_test)
    batch_size: 批大小
    
    返回:
    accuracy: 测试集准确率
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    x_test, y_test = test_data
    
    # 获取输入和输出维度
    input_size = x_test.shape[1]
    if y_test.ndim > 1:
        output_size = y_test.shape[1]
    else:
        output_size = np.max(y_test) + 1  # 假设标签从0开始
    
    # 尝试从模型文件中加载模型结构信息
    model_config_path = os.path.join(os.path.dirname(model_path), 'model_config.pkl')
    if os.path.exists(model_config_path):
        with open(model_config_path, 'rb') as f:
            model_config = pickle.load(f)
            print("从配置文件加载模型结构...")
    else:
        # 如果没有配置文件，使用默认值
        model_config = {
            'input_size': input_size,
            'hidden1_size': 512,
            'hidden2_size': 64,
            'output_size': output_size,
            'activation': 'relu',
            'weight_decay': 5e-4,
            'dropout': 0.4
        }
        print("使用默认模型结构...")
    
    # 创建模型
    model = ThreeLayerNet(
        input_size=model_config['input_size'],
        hidden1_size=model_config['hidden1_size'],
        hidden2_size=model_config['hidden2_size'],
        output_size=model_config['output_size'],
        activation=model_config['activation'],
        weight_decay=model_config['weight_decay'],
        dropout=model_config['dropout']
    )
    
    # 加载模型参数
    model.load_model(model_path)
    
    # 创建测试数据迭代器
    test_iterator = BatchIterator(x_test, y_test, batch_size=batch_size, shuffle=False)
    
    # 设置为评估模式
    model.eval()
    
    total_acc = 0
    sample_count = 0
    
    # 使用no_grad上下文管理器禁用梯度计算
    with model.no_grad():
        for batch_x, batch_y in test_iterator:
            # 前向传播
            y_pred = model.forward(batch_x)
            
            # 计算准确率
            pred_classes = np.argmax(y_pred, axis=1)
            true_classes = np.argmax(batch_y, axis=1)
            acc = np.mean(pred_classes == true_classes)
            
            # 获取当前批次的样本数
            current_batch_size = len(batch_x)
            
            # 累计加权准确率
            total_acc += acc * current_batch_size
            sample_count += current_batch_size
    
    # 计算平均准确率
    accuracy = total_acc / sample_count
    
    return accuracy

def main():
    """主函数：加载已训练模型并在测试集上评估性能"""
    print("加载CIFAR-10测试数据...")
    
    # 只加载测试数据
    _, _, (x_test, y_test) = load_cifar10_data(
        normalize=True, flatten=True, one_hot=True, validation_size=0)
    
    print(f"测试集: {x_test.shape[0]}个样本, 特征维度: {x_test.shape[1]}")
    
    # 模型路径
    model_path = './models/best_model.pkl'

    # 加载模型并测试
    test_acc = load_and_test_model(model_path, (x_test, y_test))
    print(f"测试集准确率: {test_acc:.4f}")
    
if __name__ == "__main__":
    main() 