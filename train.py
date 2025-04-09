import numpy as np
import os
import time
import pickle
import argparse
import matplotlib.pyplot as plt
from src.model import ThreeLayerNet
from src.optimizer import SGD, ExponentialScheduler
from src.data_utils import load_cifar10_data, BatchIterator

def train_model(model, train_data, val_data, optimizer, batch_size=128, num_epochs=10, 
               save_best=True, seed=42, model_dir='./models', lr_decay=0.95, print_frequency=20):
    """
    训练模型
    
    参数:
    model: 神经网络模型
    train_data: 训练数据元组 (x_train, y_train)
    val_data: 验证数据元组 (x_val, y_val)
    optimizer: 优化器对象
    batch_size: 批大小
    num_epochs: 训练迭代次数
    save_best: 是否保存最佳模型
    seed: 随机种子，确保结果可复现
    model_dir: 模型和结果保存目录
    lr_decay: 学习率衰减系数
    print_frequency: 打印频率
    
    返回:
    训练历史记录
    """
    # 设置随机种子确保可复现
    np.random.seed(seed)
    
    x_train, y_train = train_data
    x_val, y_val = val_data
    
    # 创建批处理迭代器
    train_iterator = BatchIterator(x_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
    val_iterator = BatchIterator(x_val, y_val, batch_size=batch_size, shuffle=False, seed=seed)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': [],
        'best_epoch': 0,
        'best_val_acc': 0.0
    }
    
    # 保存最佳模型的相关变量
    best_val_acc = 0
    best_model_params = None
    best_epoch = 0
    
    # 确保模型保存目录存在
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # 初始学习率
    initial_lr = optimizer.learning_rate
    
    # 创建学习率调度器
    total_steps = train_iterator.num_batches * num_epochs
    scheduler = ExponentialScheduler(
        initial_lr=initial_lr,
        stage_length=train_iterator.num_batches,
        decay=lr_decay,
        staircase=True
    )
    
    # 记录训练开始时间
    start_time = time.time()
    
    print("Starting training...")
    print(f"Training set size: {x_train.shape[0]}, Number of batches: {train_iterator.num_batches}, Batch size: {batch_size}")
    print(f"Initial learning rate: {initial_lr}")
    print(f"Learning rate schedule: Each epoch multiply by {lr_decay}")
    print()
    
    # 总迭代次数计数
    total_iter = 0
    
    for epoch in range(num_epochs):
        # 每个epoch开始时重置计数器
        epoch_start_time = time.time()
        total_loss = 0
        total_acc = 0
        batch_count = 0
        sample_count = 0
            
        # 记录当前epoch的学习率
        history['learning_rates'].append(optimizer.learning_rate)
        
        # 设置模型为训练模式
        model.train()
        
        # 训练一个epoch
        for batch_x, batch_y in train_iterator:
            total_iter += 1
            
            # 更新学习率
            optimizer.learning_rate = scheduler(total_iter)
            
            # 每个batch前清零梯度
            model.zero_grad()
            
            # 前向传播
            y_pred = model.forward(batch_x)
            
            # 计算损失
            loss = model.loss(batch_y, y_pred)
            
            # 反向传播
            model.backward(batch_y, y_pred)
            
            # 更新参数
            optimizer.update(model)
            
            # 计算准确率
            pred_classes = np.argmax(y_pred, axis=1)
            true_classes = np.argmax(batch_y, axis=1)
            acc = np.mean(pred_classes == true_classes)
            
            # 获取当前批次的样本数
            current_batch_size = len(batch_x)
            
            # 累计加权损失和准确率
            total_loss += loss * current_batch_size
            total_acc += acc * current_batch_size
            sample_count += current_batch_size
            batch_count += 1
            
            # 打印进度
            if batch_count % print_frequency == 0:
                print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_count}/{train_iterator.num_batches}, "
                      f"Loss: {loss:.4f}, Accuracy: {acc:.4f}, LR: {optimizer.learning_rate:.6f}")
        
        # 计算加权平均损失和准确率
        avg_loss = total_loss / sample_count
        avg_acc = total_acc / sample_count
        
        # 评估验证集性能
        val_loss, val_acc = evaluate_model(model, val_iterator)
        
        # 记录训练历史
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(avg_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 计算epoch耗时
        epoch_time = time.time() - epoch_start_time
        
        # 打印epoch总结
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s - "
              f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        # 保存最佳模型
        if save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_params = model.get_params()
            best_model_path = os.path.join(model_dir, 'best_model.pkl')
            model.save_model(best_model_path)
            print(f"New best model saved to {best_model_path}, validation accuracy: {val_acc:.4f} at epoch {epoch+1}")
            
            # 更新历史记录中的最佳模型信息
            history['best_epoch'] = best_epoch
            history['best_val_acc'] = best_val_acc
    
    # 训练结束，记录总训练时间
    total_time = time.time() - start_time
    print()
    print(f"Training completed, total time: {total_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    
    # 如果有最佳模型，恢复它
    if best_model_params:
        model.set_params(best_model_params)
        print(f"Best model restored, validation accuracy: {best_val_acc:.4f} from epoch {best_epoch}")
        
    # 可视化训练过程
    plot_training_history(history, model_dir=model_dir)
    
    return history

def evaluate_model(model, data_iterator):
    """
    评估模型在给定数据迭代器上的性能
    
    参数:
    model: 神经网络模型
    data_iterator: 数据迭代器
    
    返回:
    (loss, accuracy)元组
    """
    # 设置为评估模式
    model.eval()
    
    total_loss = 0
    total_acc = 0
    sample_count = 0
    
    # 使用no_grad上下文管理器禁用梯度计算
    with model.no_grad():
        for batch_x, batch_y in data_iterator:
            # 前向传播
            y_pred = model.forward(batch_x)
            
            # 计算损失
            loss = model.loss(batch_y, y_pred)
            
            # 计算准确率
            pred_classes = np.argmax(y_pred, axis=1)
            true_classes = np.argmax(batch_y, axis=1)
            acc = np.mean(pred_classes == true_classes)
            
            # 获取当前批次的样本数
            current_batch_size = len(batch_x)
            
            # 累计加权损失和准确率
            total_loss += loss * current_batch_size
            total_acc += acc * current_batch_size
            sample_count += current_batch_size
    
    # 计算平均损失和准确率
    avg_loss = total_loss / sample_count
    avg_acc = total_acc / sample_count
    
    return avg_loss, avg_acc

def plot_training_history(history, model_dir='./models', dpi=150):
    """
    可视化训练历史
    
    参数:
    history: 训练历史记录
    model_dir: 图像保存目录
    dpi: 图像分辨率
    """
    # 确保保存目录存在
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-', label='Training Loss')
    plt.plot(history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 使用紧凑布局
    plt.tight_layout()
    # 保存图像
    plt.savefig(os.path.join(model_dir, 'training_history.png'), dpi=dpi)
    plt.close()

def visualize_weights(model, model_dir='./models', dpi=150):
    """
    可视化模型权重
    
    参数:
    model: 神经网络模型
    model_dir: 模型保存目录的根目录
    dpi: 图像分辨率
    """
    print("Generating weight visualization...")
    
    # 创建一个目录保存可视化结果
    vis_dir = os.path.join(model_dir, 'weights_visualization')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # 获取第一层权重 (FC1)
    W1 = model.layers['fc1'].params['W']
    
    # 对于CIFAR-10，输入是3072维 (32x32x3)
    # 将权重reshape为图像尺寸 (32x32x3) 便于可视化
    n_neurons = min(100, W1.shape[1])  # 最多显示100个神经元的权重
    
    # 计算网格大小 (为了得到接近方形的排列)
    grid_size = int(np.ceil(np.sqrt(n_neurons)))
    
    # 创建大图
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    fig.suptitle('First Layer Neuron Weights', fontsize=16)
    
    # 重塑每个神经元的权重为32x32x3的图像
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < n_neurons:
                # 获取第idx个神经元的权重
                w = W1[:, idx].reshape(32, 32, 3)
                
                # 归一化权重以便更好地可视化
                w_min, w_max = w.min(), w.max()
                if w_max > w_min:
                    w = (w - w_min) / (w_max - w_min)
                
                # 显示权重
                axes[i, j].imshow(w)
                axes[i, j].axis('off')
                axes[i, j].set_title(f'Neuron {idx+1}')
            else:
                axes[i, j].axis('off')
    
    # 调整布局，为标题留出空间
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # 保存高分辨率图像
    plt.savefig(os.path.join(vis_dir, 'layer1_weights.png'), dpi=dpi)
    plt.close()
    
    # 可视化第二层权重的模式
    W2 = model.layers['fc2'].params['W']
    
    # 可视化权重矩阵的热力图
    plt.figure(figsize=(12, 10))
    plt.imshow(W2, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Weights from Hidden Layer 1 to Hidden Layer 2')
    plt.savefig(os.path.join(vis_dir, 'layer2_weights.png'), dpi=dpi)
    plt.close()
    
    # 可视化第三层权重的模式
    W3 = model.layers['fc3'].params['W']
    
    plt.figure(figsize=(10, 8))
    plt.imshow(W3, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Weights from Hidden Layer 2 to Output Layer')
    plt.savefig(os.path.join(vis_dir, 'layer3_weights.png'), dpi=dpi)
    plt.close()
    
    print(f"Weight visualization saved to {vis_dir} directory")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train a 3-layer neural network on CIFAR-10')
    
    # 模型结构参数
    parser.add_argument('--hidden1', type=int, default=512, help='Size of first hidden layer (default: 512)')
    parser.add_argument('--hidden2', type=int, default=64, help='Size of second hidden layer (default: 64)')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid'], 
                        help='Activation function (default: relu)')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs (default: 15)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer (default: 0.9)')
    parser.add_argument('--lr_decay', type=float, default=0.95, 
                        help='Learning rate decay factor per epoch (default: 0.95)')
    
    # 正则化参数
    parser.add_argument('--weight_decay', type=float, default=5e-4, 
                        help='L2 regularization parameter (default: 5e-4)')
    parser.add_argument('--dropout', type=float, default=0.0, 
                        help='Dropout rate, 0 to disable dropout (default: 0.0)')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no_save_best', action='store_true', help='Do not save best model during training')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory to save models (default: ./models)')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for saved figures (default: 150)')
    parser.add_argument('--print_frequency', type=int, default=100, help='Print frequency during training (default: 100)')
    
    return parser.parse_args()

def main():
    """
    主函数：加载数据，创建模型，训练模型并评估
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置全局随机种子
    np.random.seed(args.seed)
    print("\n" + "="*60)
    print("训练开始")
    print("="*60)
    print(f"随机种子: {args.seed}")
    
    # 创建模型目录
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    # 加载CIFAR-10数据
    print("\n" + "-"*60)
    print("数据加载")
    print("-"*60)
    print("正在加载CIFAR-10数据...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data(
        normalize=True, 
        flatten=True, 
        one_hot=True, 
        validation_size=0.1, 
        seed=args.seed)
    
    print(f"训练集: {x_train.shape[0]} 样本")
    print(f"验证集: {x_val.shape[0]} 样本")
    print(f"测试集: {x_test.shape[0]} 样本")
    print(f"输入特征维度: {x_train.shape[1]}")
    
    # 创建模型
    input_size = x_train.shape[1]  # 输入特征的维度
    output_size = 10  # 10 classes in CIFAR-10
    
    print("\n" + "-"*60)
    print("模型创建")
    print("-"*60)
    print(f"模型架构: {input_size} -> {args.hidden1} -> {args.hidden2} -> {output_size}")
    print(f"激活函数: {args.activation}")
    print(f"权重衰减: {args.weight_decay}")
    print(f"Dropout率: {args.dropout}")
    
    model = ThreeLayerNet(
        input_size=input_size,
        hidden1_size=args.hidden1,
        hidden2_size=args.hidden2,
        output_size=output_size,
        activation=args.activation,
        weight_decay=args.weight_decay,
        dropout=args.dropout
    )
    
    # 保存模型结构信息
    model_config = {
        'input_size': input_size,
        'hidden1_size': args.hidden1,
        'hidden2_size': args.hidden2,
        'output_size': output_size,
        'activation': args.activation,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout
    }
    
    config_path = os.path.join(args.model_dir, 'model_config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(model_config, f)
    print(f"模型结构已保存到: {config_path}")
    
    # 创建优化器
    print("\n" + "-"*60)
    print("优化器设置")
    print("-"*60)
    print(f"优化器: SGD")
    print(f"初始学习率: {args.lr}")
    print(f"动量: {args.momentum}")
    print(f"学习率衰减: {args.lr_decay}")
    
    optimizer = SGD(
        learning_rate=args.lr,
        momentum=args.momentum
    )
    
    # 训练模型
    print("\n" + "-"*60)
    print("开始训练")
    print("-"*60)
    print(f"训练集大小: {x_train.shape[0]}, 批量大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"打印频率: 每{args.print_frequency}批次")
    print("-"*60)

    history = train_model(
        model=model,
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
        optimizer=optimizer,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        save_best=not args.no_save_best,
        seed=args.seed,
        model_dir=args.model_dir,
        lr_decay=args.lr_decay,
        print_frequency=args.print_frequency
    )
    
    # 可视化模型权重
    print("\n" + "-"*60)
    print("模型可视化")
    print("-"*60)
    visualize_weights(model, model_dir=args.model_dir, dpi=args.dpi)
    
    # 保存训练历史
    history_path = os.path.join(args.model_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"训练历史已保存到: {history_path}")
    
    # 输出训练结果摘要
    print("\n" + "="*60)
    print("训练结果摘要")
    print("="*60)
    print(f"最佳验证准确率: {history['best_val_acc']:.4f} (第 {history['best_epoch']} 轮)")
    print(f"最终训练准确率: {history['train_acc'][-1]:.4f}")
    print(f"最终验证准确率: {history['val_acc'][-1]:.4f}")
    print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
    print(f"最终验证损失: {history['val_loss'][-1]:.4f}")
    
    # 输出模型参数信息
    print("\n" + "="*60)
    print("模型参数信息")
    print("="*60)
    print(f"隐藏层大小: [{args.hidden1}, {args.hidden2}]")
    print(f"激活函数: {args.activation}")
    print(f"批量大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"动量: {args.momentum}")
    print(f"权重衰减: {args.weight_decay}")
    print(f"Dropout率: {args.dropout}")
    print(f"学习率衰减: {args.lr_decay}")
    print(f"随机种子: {args.seed}")
    
    print("\n" + "="*60)
    print("训练完成")
    print("="*60)

if __name__ == "__main__":
    main() 