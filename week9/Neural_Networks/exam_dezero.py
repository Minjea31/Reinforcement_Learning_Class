import numpy as np
from dezero import Variable
import dezero.functions as F

x_np = np.array(5.0)
x = Variable(x_np)

y = 3 * x ** 2
print(y)

y.backward()
print(x.grad)

a = np.array([1,2,3])
b = np.array([4,5,6])
a,b = Variable(a), Variable(b)
c= F.matmul(a,b)
print(c)

a = np.array([[1,2],[3,4]])
b = np.array([[5,6], [7,8]])
c = F.matmul(a,b)
print(c)

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2)**2 + (x0 -1)**2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

iters = 10000 # 반복횟수
lr = 0.001 #학습률

for i in range(iters):
    y = rosenbrock(x0, x1)

    # 이전 반복에서 더해진 미분 초기화
    x0.cleargrad()
    x1.cleargrad()

    y.backward()

    x0.data -= lr * x0.grad.data
    x1.data -= lr * x1.grad.data

    print(x0, x1)
print(x0,x1)

# y = rosenbrock(x0, x1)
# y.backward()
# print(x0.grad, x1.grad)