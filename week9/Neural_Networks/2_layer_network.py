import numpy as np
from numpy.random import randn

N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N,D_in), randn(N, D_out) # x = (64, 1000), y= (64 * 10)
w1, w2 = randn(D_in, H), randn(H, D_out) # w1 = (1000, 100), w2 = (100, 10)

for t in range(2000):
    h = 1 / (1+np.exp(-x.dot(w1))) # 입력과 가중치를 곱해서 활성화 합수에 입력 (64, 100)
    # sigmoid = 1 / (1+ e^(-x))
    y_pred = h.dot(w2) # y 예측값 = h * w2
    loss = np.square(y_pred - y).sum()
    # 여기서는 loss 함수가 (y^ - y)^2 임
    print(t, loss)

    grad_y_pred = 2 * (y_pred - y) # ∂E / ∂y_pred
    grad_w2 = h.T.dot(grad_y_pred) # ∂E / w2 = (∂E / ∂y_pred) * (∂y_pred / ∂w2)
    # h.T인 이유 : w2의 가중치 하나에 대해, 64개 샘플이 각각 만든 기울기를 모두 더해야 하기 때문이야. 이때 샘플이 1개의 열임.
    grad_h = grad_y_pred.dot(w2.T)
    # 이것도 y^1 을 행렬식으로 써보고 유도하면 w2가 .T 되서 곱해지는걸 알수있음.
    grad_w1 = x.T.dot(grad_h * h * (1-h))

    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2