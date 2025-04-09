# 基于NumPy实现的CIFAR-10三层神经网络分类器

本项目使用NumPy从零实现了MLP，用于CIFAR-10分类，不依赖任何深度学习框架，手动实现了前向传播和反向传播。

## 项目结构

```
.
├── src/                         # 源代码目录
│   ├── __init__.py              # 包初始化文件
│   ├── activation.py            # 激活函数实现
│   ├── data_utils.py            # 数据加载和处理工具
│   ├── layers.py                # 神经网络层实现
│   ├── loss.py                  # 损失函数实现
│   ├── model.py                 # 三层神经网络模型实现
│   └── optimizer.py             # 优化器实现
├── train.py                     # 训练脚本
├── test.py                      # 测试脚本
├── hyperparameter_tuning.py     # 超参数网格搜索脚本
├── read_pickle.ipynb            # 读取pickle文件工具
├── data/                        # 数据存放目录
├── models/                      # 最优模型存放目录、训练结果可视化、模型参数可视化
├── logs/                        # 训练日志和超参数网格搜索结果
└── README.md                    # 项目说明
```

## 数据准备

本项目使用CIFAR-10数据集进行图像分类。数据会在首次运行`train.py`时自动下载到`data`目录下。如果数据已存在，将直接使用现有数据。  
也可以手动将百度网盘中的数据拷贝至`data`目录下。

## 模型训练

`train.py`主要命令行参数说明：
- `--hidden1`: 第一个隐藏层神经元数量，默认为512
- `--hidden2`: 第二个隐藏层神经元数量，默认为64
- `--activation`: 激活函数类型，可选'relu'或'sigmoid'，默认为'relu'
- `--lr`: 初始学习率，默认为0.001
- `--batch_size`: 批次大小，默认为64
- `--epochs`: 训练轮数，默认为15
- `--lr_decay`: 学习率衰减系数，默认为0.95
- `--dropout`: dropout比率，默认为0.0
- `--weight_decay`: L2正则化强度，默认为5e-4
- `--seed`: 随机种子，默认为42
- `--model_dir`: 模型保存目录，默认为'./models'  
- '--print_frequency': 训练结果打印频率，每多少个batch打印一次结果，默认为100

---
**最优模型的训练，使用默认参数直接运行**
```bash
python train.py
```

也可以通过命令行参数自定义训练配置，例如：
```bash
# 使用更大的隐藏层和更多的训练轮数
python train.py --hidden1 1024 --hidden2 128 --epochs 20

# 使用sigmoid激活函数和更高的学习率
python train.py --activation sigmoid --lr 0.01

# 使用更大的批次大小和更强的正则化
python train.py --batch_size 128 --dropout 0.5 --weight_decay 1e-3

# 自定义多个参数
python train.py --hidden1 1024 --hidden2 128 --activation relu --lr 0.001 --batch_size 128 --epochs 20 --dropout 0.5
```

训练过程中会自动保存最佳模型到`models`目录，并生成训练和验证的Loss、Acc曲线，以及最优模型参数可视化结果。

## 模型测试

要测试模型，有两种方案：  
1. 把网盘中的`best_model.pkl`和`model_config.pkl`拷贝至`models`文件夹下，然后运行`test.py`
2. 使用训练脚本`train.py`训练出自己的模型，进行测试

使用以下命令在测试集上评估训练好的模型性能：

```bash
python test.py
```

测试脚本会自动加载`models/best_model.pkl`模型权重和`models/model_config.pkl`模型超参数`meta`文件，并在CIFAR-10测试集上计算准确率。  

## 超参数网格搜索

使用以下命令进行超参数网格搜索：

```bash
python hyperparameter_tuning.py
```

该脚本会尝试不同的超参数组合，并记录各组合下的验证集性能，帮助找到最优的超参数设置。  
注意：因时间限制，只搜索了部分超参数组合，**最终的最优模型的参数，是手动调参得出的最优结果，并非网格搜索得出**。

## 模型权重和数据下载

训练好的模型权重文件已上传到百度网盘，下载地址：  
https://pan.baidu.com/s/1zwlKVVZImy7rnNA-bZ6QwQ?pwd=0042  
提取码: 0042 

CIFAR-10数据也一同上传到了百度网盘，链接同上

## 实验报告

关于MLP前向传播、反向传播、交叉熵loss计算、ReLU激活函数、SGD优化器、学习率指数衰减等Numpy手动实现，请见具体代码和实验报告

关于具体模型结构请见实验报告

关于具体实验结果和可视化曲线、模型参数等，请见实验报告

## 训好的模型参数、具体实验结果等可直接通过pickle查看

具体查看`read_pickle.ipynb`文件