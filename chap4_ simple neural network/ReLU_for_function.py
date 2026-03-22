import numpy as np
import matplotlib.pyplot as plt

# 目标函数
def target_function(x):
    return 0.5*np.sin(5*x) + 0.3*np.cos(3*x) + 0.2*x**2 - 0.1*x

# 数据
np.random.seed(42)

x_train = np.random.uniform(-2, 2, 2000).reshape(-1,1)
y_train = target_function(x_train)

x_test = np.linspace(-2, 2, 500).reshape(-1,1)
y_test = target_function(x_test)

# 输入、输出归一化
x_train = x_train / 2
x_test = x_test / 2

y_scale = np.max(np.abs(y_train))
y_train = y_train / y_scale

# 网络结构
input_dim = 1
h1 = 128
h2 = 128

# He初始化
W1 = np.random.randn(input_dim, h1) * np.sqrt(2/input_dim)
b1 = np.zeros((1, h1))

W2 = np.random.randn(h1, h2) * np.sqrt(2/h1)
b2 = np.zeros((1, h2))

W3 = np.random.randn(h2, 1) * np.sqrt(2/h2)
b3 = np.zeros((1, 1))

# 参数
lr = 0.001
beta1, beta2 = 0.9, 0.999
eps = 1e-8

def init_adam(param):
    return np.zeros_like(param), np.zeros_like(param)

params = [W1, b1, W2, b2, W3, b3]
grads = [np.zeros_like(p) for p in params]
ms = [np.zeros_like(p) for p in params]
vs = [np.zeros_like(p) for p in params]

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

# 训练
epochs = 6000
loss_history = []

for t in range(1, epochs+1):

    # forward
    z1 = x_train @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    z3 = a2 @ W3 + b3

    # 残差
    y_pred = z3 + x_train

    # loss
    loss = np.mean((y_pred - y_train)**2)
    loss_history.append(loss)

    # backward
    dy = 2*(y_pred - y_train)/len(y_train)

    dz3 = dy
    dW3 = a2.T @ dz3
    db3 = np.sum(dz3, axis=0, keepdims=True)

    da2 = dz3 @ W3.T
    dz2 = da2 * relu_grad(z2)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_grad(z1)
    dW1 = x_train.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    grads = [dW1, db1, dW2, db2, dW3, db3]

    # Adam更新
    for i in range(len(params)):
        ms[i] = beta1*ms[i] + (1-beta1)*grads[i]
        vs[i] = beta2*vs[i] + (1-beta2)*(grads[i]**2)

        m_hat = ms[i] / (1-beta1**t)
        v_hat = vs[i] / (1-beta2**t)

        params[i] -= lr * m_hat / (np.sqrt(v_hat)+eps)

    if t % 500 == 0:
        print(f"Epoch {t}, Loss: {loss:.6f}")

#测试
z1 = x_test @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
a2 = relu(z2)
z3 = a2 @ W3 + b3

y_pred_test = (z3 + x_test) * y_scale  # 还原

# 误差评估指标
y_true = y_test.flatten()
y_pred = y_pred_test.flatten()

# MSE（均方误差）
mse = np.mean((y_true - y_pred)**2)
# MAE（平均绝对误差）
mae = np.mean(np.abs(y_true - y_pred))
# R^2（决定系数）
ss_res = np.sum((y_true - y_pred)**2)
ss_tot = np.sum((y_true - np.mean(y_true))**2)
r2 = 1 - ss_res / ss_tot

print("\n===== Evaluation Metrics =====")
print(f"MSE : {mse:.6f}")
print(f"MAE : {mae:.6f}")
print(f"R^2 : {r2:.6f}")

plt.figure(figsize=(8,5))
plt.plot(x_test*2, y_test, label="True")
plt.plot(x_test*2, y_pred_test, label="Prediction")
plt.legend()
plt.title("Deep ReLU (Only x Input)")
plt.show()

plt.figure()
plt.plot(loss_history)
plt.title("Loss Curve")
plt.show()