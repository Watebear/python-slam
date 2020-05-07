import numpy as np

# 创建一个 2x3 的矩阵
A = np.mat([[1, 2, 3], [4, 5, 6]])
# 创建一个 1x2 的向量
b = np.array([1, 2, 3])
# 创建随机数矩阵
C = np.random.randint(1, 10, (3, 3))

print('A is 2x3 :\n', A)
print('b is 1x3 :', b)
print('C is 3x3 :\n', C)

# 转置
print('A.T:\n', A.T)
# 求各元素之和
print(np.sum(A))
print(np.sum(A, axis=0))
print(np.sum(A, axis=1))
# 求迹
print('tr(A):', np.trace(A))
# 数乘
print(2 * A)
# 求逆矩阵
print(np.linalg.inv(C))
# 求行列式
print(np.linalg.det(C))
# 求特征值
eigvalue, eigvector = np.linalg.eig(C)
print("eigvalue:", eigvalue)
print("eigvector:", eigvector)

# Cx=b
x = np.linalg.solve(C, b)
print('x:', x)
