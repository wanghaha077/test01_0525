import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from logistic_regression import generate_synthetic_data, LogisticRegression
w = np.array([2, -3.4])  # 模型参数
b = 4.2  # 偏置

# 生成数据集
X, y = generate_synthetic_data(w, b, num_samples=50000)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for learning_rate in [0.1, 0.01,0.001]:
    for num_iterations in [10000, 30000]:
        # 训练模型
        model = LogisticRegression(learning_rate, num_iterations)
        loss_history = model.fit(X_train, y_train)

        # 测试模型
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {accuracy * 100:.2f}%')

        # 4. 可视化损失变化
        plt.plot(loss_history)
        plt.xlabel(f'Iterations (Total: {num_iterations}),learning_rate{learning_rate}')  # 使用格式化字符串
        plt.ylabel('Loss')
        plt.title('Loss over iterations')
        plt.show()
