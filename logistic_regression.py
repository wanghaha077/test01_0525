import numpy as np

# 1. 生成合成数据集
def generate_synthetic_data(w, b, num_samples=100):
    X = np.random.rand(num_samples, len(w))  # 生成随机特征矩阵
    e = np.random.randn(num_samples) * 0.1  # 添加噪声
    y = (X @ w + b + e > 3).astype(int)  # 根据线性关系生成标签
    return X, y


# 2. 定义对数几率回归模型
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):#确定学习率和迭代次数
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)  # 初始化权重
        self.b = 0  # 初始化偏置

        loss_history = []
        for i in range(self.num_iterations):
            linear_model = np.dot(X, self.w) + self.b
            y_predicted = self.sigmoid(linear_model)

            # 计算损失（交叉熵损失）
            loss = - (1 / num_samples) * np.sum(
                y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15))
            loss_history.append(loss)

            # 计算梯度
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            if (i + 1) % 10000 == 0:  # 每10次迭代输出
                print(f"Iteration {i + 1}, Loss: {loss:.4f}")

            # 更新参数
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

        return loss_history

    def predict(self, X):
        linear_model = np.dot(X, self.w) + self.b
        y_predicted = self.sigmoid(linear_model)
        return (y_predicted > 0.5).astype(int)



