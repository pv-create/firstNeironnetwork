import numpy as np

INPUT_DIM = 4
OUT_DIM = 3
H_DIM = 5

x = np.random.randn(INPUT_DIM)

w1 = np.random.randn(INPUT_DIM, H_DIM)
b1 = np.random.randn(H_DIM)
w2 = np.random.randn(H_DIM, OUT_DIM)
b2 = np.random.randn(OUT_DIM)


def relu(t):
    return np.maximum(t, 0)


def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)


def predict(x):
    t1 = x @ w1
    h1 = relu(t1)
    t2 = h1 @ w2 + b2
    z = softmax(t2)
    return z


probs = predict(x)
pred_class = np.argmax(probs)
class_names = ['sites', 'vermistor', 'verginika']
print(class_names[pred_class])


# моя первая попытка написать нейронную сеть
# нейронная сеть решает задачу класификации
# дано три класса
# на вход подается 4 сигнала
# количество нейронов на втором уровне равно 5
# все веса и значения для входа задаются рандомно
