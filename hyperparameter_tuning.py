import numpy as np
import os
import time
import pickle
import argparse
from src.model import ThreeLayerNet
from src.optimizer import SGD, ExponentialScheduler
from src.data_utils import load_cifar10_data, BatchIterator

def train_and_evaluate(params, train_data, val_data, num_epochs=6, batch_size=64, 
                      early_stopping_patience=3):
    """
    训练模型并评估性能
    
    参数:
    params: 参数字典
    train_data: 训练数据元组 (x_train, y_train)
    val_data: 验证数据元组 (x_val, y_val)
    num_epochs: 训练迭代次数
    batch_size: 批大小
    early_stopping_patience: 早停耐心值
    
    返回:
    验证准确率
    """
    x_train, y_train = train_data
    x_val, y_val = val_data
    
    # 创建模型
    input_size = x_train.shape[1]
    output_size = y_train.shape[1] if y_train.ndim > 1 else np.max(y_train) + 1
    
    model = ThreeLayerNet(
        input_size=input_size,
        hidden1_size=params['hidden1_size'],
        hidden2_size=params['hidden2_size'],
        output_size=output_size,
        activation=params['activation'],
        weight_decay=params['weight_decay'],
        dropout=params['dropout']
    )
    
    # 创建优化器
    optimizer = SGD(
        learning_rate=params['learning_rate'],
        momentum=0.9
    )
    
    # 创建学习率调度器
    scheduler = ExponentialScheduler(
        initial_lr=params['learning_rate'],
        stage_length=len(x_train) // batch_size,
        decay=0.95,
        staircase=True
    )
    
    # 训练变量
    best_val_acc = 0
    patience_counter = 0
    best_model_params = None
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练一个epoch
        model.train()  # 设置为训练模式
        train_iterator = BatchIterator(x_train, y_train, batch_size=batch_size, shuffle=True)
        
        for batch_x, batch_y in train_iterator:
            # 更新学习率
            optimizer.learning_rate = scheduler(epoch)
            
            # 梯度清零
            model.zero_grad()
            
            # 前向传播
            y_pred = model.forward(batch_x)
            
            # 计算损失
            loss = model.loss(batch_y, y_pred)
            
            # 反向传播
            model.backward(batch_y, y_pred)
            
            # 更新参数
            optimizer.update(model)
        
        # 评估验证集
        model.eval()  # 设置为评估模式
        val_loss, val_acc = evaluate_model(model, val_data, batch_size=batch_size)
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_params = model.get_params()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 如果达到早停条件，提前结束训练
        if patience_counter >= early_stopping_patience:
            break
    
    # 恢复最佳模型
    if best_model_params:
        model.set_params(best_model_params)
    
    return best_val_acc

def evaluate_model(model, data, batch_size=128):
    """评估模型性能"""
    x, y = data
    
    if len(x) > batch_size:
        iterator = BatchIterator(x, y, batch_size=batch_size, shuffle=False)
        total_loss = 0
        total_acc = 0
        sample_count = 0
        
        # 使用no_grad上下文管理器禁用梯度计算
        with model.no_grad():
            for batch_x, batch_y in iterator:
                # 前向传播
                y_pred = model.forward(batch_x)
                
                # 计算损失
                loss = model.loss(batch_y, y_pred)
                
                # 计算准确率
                pred_classes = np.argmax(y_pred, axis=1)
                true_classes = np.argmax(batch_y, axis=1)
                acc = np.mean(pred_classes == true_classes)
                
                batch_size = len(batch_x)
                total_loss += loss * batch_size
                total_acc += acc * batch_size
                sample_count += batch_size
        
        avg_loss = total_loss / sample_count
        avg_acc = total_acc / sample_count
    else:
        # 使用no_grad上下文管理器禁用梯度计算
        with model.no_grad():
            # 前向传播
            y_pred = model.forward(x)
            
            # 计算损失
            avg_loss = model.loss(y, y_pred)
            
            # 计算准确率
            pred_classes = np.argmax(y_pred, axis=1)
            true_classes = np.argmax(y, axis=1)
            avg_acc = np.mean(pred_classes == true_classes)
    
    return avg_loss, avg_acc

def generate_report(results, logs_dir):
    """生成详细的超参数搜索报告"""
    # 按验证准确率排序
    sorted_results = sorted(results, key=lambda x: x['val_acc'], reverse=True)
    
    # 创建报告文件
    report_path = os.path.join(logs_dir, 'hyperparameter_search_report.txt')
    with open(report_path, 'w') as f:
        f.write("超参数搜索报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 写入最佳参数
        f.write("最佳参数组合:\n")
        f.write("-" * 30 + "\n")
        for key, value in sorted_results[0]['params'].items():
            f.write(f"{key}: {value}\n")
        f.write(f"验证准确率: {sorted_results[0]['val_acc']:.4f}\n\n")
        
        # 写入所有参数组合的性能
        f.write("所有参数组合的性能 (按验证准确率降序排列):\n")
        f.write("-" * 30 + "\n")
        for i, result in enumerate(sorted_results):
            f.write(f"\n组合 {i+1}:\n")
            for key, value in result['params'].items():
                f.write(f"{key}: {value}\n")
            f.write(f"验证准确率: {result['val_acc']:.4f}\n")
    
    print(f"详细报告已保存到 {report_path}")
    
    # 打印前5个最佳结果
    print("\n前5个最佳参数组合:")
    print("-" * 30)
    for i, result in enumerate(sorted_results[:5]):
        print(f"\n组合 {i+1}:")
        for key, value in result['params'].items():
            print(f"{key}: {value}")
        print(f"验证准确率: {result['val_acc']:.4f}")

def main():
    """主函数：执行网格搜索"""
    parser = argparse.ArgumentParser(description='Grid search for hyperparameter tuning')
    # 节省时间，超参数搜索时，只训练6个epoch
    parser.add_argument('--num_epochs', type=int, default=6, help='Number of epochs per trial')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--logs_dir', type=str, default='./logs/parameter', help='Directory to save models')
    args = parser.parse_args()
    
    print("Loading CIFAR-10 data...")
    (x_train, y_train), (x_val, y_val), _ = load_cifar10_data(
        normalize=True, flatten=True, one_hot=True, validation_size=0.1)
    
    # 定义参数搜索范围
    # 节省时间，减少了实际超参数搜索范围
    param_grid = {
        'hidden1_size': [128, 256],               # 隐藏层1大小 [128, 256, 512]
        'hidden2_size': [64, 128],                # 隐藏层2大小 [64, 128, 256]
        'activation': ['relu'],                   # 激活函数 ['relu', 'sigmoid']
        'learning_rate': [0.001, 0.0001],          # 学习率 [0.01, 0.001, 0.0001]
        'weight_decay': [0.001, 0.0001],          # L2正则化系数 [0.0, 0.0001, 0.001]
        'dropout': [0.3, 0.0],               # dropout率 [0.0, 0.3, 0.4, 0.5]
        'batch_size': [32, 64]                   # 批次大小 [32, 64, 128] 
    }
    
    # 记录开始时间
    start_time = time.time()
    
    # 存储所有试验结果
    results = []
    best_val_acc = 0
    best_params = None
    
    # 生成所有参数组合
    from itertools import product
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    
    print(f"Starting grid search with {len(param_combinations)} combinations...")
    
    # 执行网格搜索
    for i, params in enumerate(param_combinations):
        print(f"\nTrial {i+1}/{len(param_combinations)}")
        print(f"Parameters: {params}")
        
        # 训练和评估
        val_acc = train_and_evaluate(
            params=params,
            train_data=(x_train, y_train),
            val_data=(x_val, y_val),
            num_epochs=args.num_epochs,
            batch_size=params['batch_size'],
            early_stopping_patience=args.early_stopping_patience
        )
        
        # 记录结果
        results.append({
            'params': params,
            'val_acc': val_acc
        })
        
        # 打印每次试验的验证准确率
        print(f"Validation accuracy: {val_acc:.4f}")
        
        # 更新最佳参数
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            print(f"New best validation accuracy: {best_val_acc:.4f}")
            print(f"Best parameters: {best_params}")
    
    # 计算总运行时间
    total_time = time.time() - start_time
    print(f"\nGrid search completed in {total_time:.2f} seconds")
    
    # 保存优化结果
    results_dict = {
        'best_params': best_params,
        'best_score': best_val_acc,
        'all_trials': results,
        'total_time': total_time
    }
    
    os.makedirs(args.logs_dir, exist_ok=True)
    results_path = os.path.join(args.logs_dir, 'grid_search_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results_dict, f)
    
    print(f"Results saved to {results_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best parameters: {best_params}")
    
    # 生成详细报告
    generate_report(results, args.logs_dir)

if __name__ == "__main__":
    main() 