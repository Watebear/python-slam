import numpy as np
import math
# 1. Representation of three rotation methods: Quateration, RMatrixT, RAxisAngle
# 2. Representation of three rotation methods: Lie_Group, Lie_Algebra

class Quateration():
    def __init__(self, s, x, y, z):
        self.s = s
        self.x = x
        self.y = y
        self.z = z

    def add(self, quater2):
        # 加法
        s = self.s + quater2.s
        x = self.x + quater2.x
        y = self.y + quater2.y
        z = self.z + quater2.z
        return Quateration(s=s,x=x,y=y,z=z)

    def sub(self, quater2):
        # 减法
        s = self.s - quater2.s
        x = self.x - quater2.x
        y = self.y - quater2.y
        z = self.z - quater2.z
        return Quateration(s=s, x=x, y=y, z=z)

    def dot(self, quater2):
        # 四元数乘法
        s = self.s*quater2.s - self.x*quater2.x - self.y*quater2.y - self.z*quater2.z
        x = self.s*quater2.x + self.x*quater2.s + self.y*quater2.z - self.z*quater2.y
        y = self.s*quater2.y - self.x*quater2.z + self.y*quater2.s + self.z*quater2.x
        z = self.s*quater2.z + self.x*quater2.y - self.y*quater2.x + self.z*quater2.s
        return Quateration(s=s, x=x, y=y, z=z)

    def multiply(self, scale):
        # 数乘
        s = self.s * scale
        x = self.x * scale
        y = self.y * scale
        z = self.z * scale
        return Quateration(s=s, x=x, y=y, z=z)

    def mod(self):
        # 模值
        return np.sqrt(np.square(self.s)+np.square(self.x)+np.square(self.y)+np.square(self.z))

    def conjugate(self):
        # 共轭
        return Quateration(s=self.s, x=-1*self.x, y=-1*self.y, z=-1*self.z)

    def inverse(self):
        # 求逆
        q_conj = self.conjugate()
        q_mod = self.mod()
        q_inverse = q_conj.multiply(1/(np.square(q_mod)))
        return q_inverse

    def normalize(self):
        # 归一化
        s = self.s / self.mod()
        x = self.x / self.mod()
        y = self.y / self.mod()
        z = self.z / self.mod()
        return Quateration(s=s, x=x, y=y, z=z)

    def world_coordinate(self, t1, p1):
        # q 是旋转四元数 t是平移 p是机器人坐标
        world = self.inverse().dot(p1.sub(t1)).dot(self)
        return world

    def vector_2_skew_symmetric_matrix(vector):
        # 反对称矩阵 a^
        return np.array([[0, -1 * vector[2], vector[1]],
                         [vector[2], 0, -1 * vector[0]],
                         [-1 * vector[1], vector[0], 0]])

    def to_RMatrixT(self):
        v = np.mat([self.x, self.y, self.z])
        R = v.T.dot(v) + np.square(self.s)*np.eye(3) + 2*self.s*self.vector_2_skew_symmetric_matrix(v) \
            + np.square(self.vector_2_skew_symmetric_matrix(v))
        return R


class RMatrixT():
    def __init__(self, R=None, t=None):
        # 验证旋转矩阵R是一个行列式为1的正交矩阵 !!!
        self.R = R
        self.t = t

    def R_2_thetavector(self):
        # 旋转矩阵 -- 轴+角
        theta = np.arccos((self.R.trace()-1)/2)
        eigvalue, eigvector = np.linalg.eig(self.R)
        print('eigenvalue:', eigvalue)
        print("eigenvector", eigvector)

        index = np.argwhere(eigvalue==1).reshape([1,])
        print('index:', index)
        if len(index) != 1:
            index = index[0]
        e_vector = eigvector[index]
        e_vector = self.normalize_vector(e_vector).reshape([-1])

        return theta, e_vector


    def normalize_vector(vector):
        # 归一化得到单位长度的方向向量
        sum_square = np.sum(np.square(vector))
        if sum_square != 0:
            vector = vector / sum_square
        else:
            assert "the sum square equals 0, can't normalized by vector/sum_square"
        return vector


class RAxisAngle():
    def __init__(self, theta=None, vector=None):
        self.theta = theta
        self.vector = vector

    def thetavector_2_R(self):
        # 罗德里格斯公式： R = COS@I + (1-cos@)nn^T + sin@ n^
        R = np.cos(self.theta)*np.eye(3) + \
            (1-np.cos(self.theta))*(self.vector.T.dot(self.vector)) + \
            np.sin(self.theta)*self.skew_symmetric_matrix(vector = self.vector)
        return R

    def skew_symmetric_matrix(self, vector):
        # 反对称矩阵 a^
        return np.array([[0, -1 * vector[0, 2], vector[0, 1]],
                         [vector[0, 2], 0, -1 * vector[0, 0]],
                         [-1 * vector[0, 1], vector[0, 0], 0]])


class SO3():
    def __init__(self, R=None, Q=None):
        self.R = R  # 旋转矩阵类
        self.Q = Q  # 四元数类
        # R 和 Q 不能同时为空
        if self.R is None:
             self.R = self.Q.to_RMatrixT()# Q 2 M

        self.M = self.log() # 反对称矩阵
        self.so3 = self.hat() # 李代数

    def log(self):
        # 反对称矩阵
        return np.log(self.R)

    def hat(self):
        return self.skew_symmetric_matrix_2_vector(M = self.M)

    def vector_2_skew_symmetric_matrix(self, vector):
        # 反对称矩阵 a^
        return np.array([[0, -1 * vector[0, 2], vector[0, 1]],
                         [vector[0, 2], 0, -1 * vector[0, 0]],
                         [-1 * vector[0, 1], vector[0, 0], 0]])

    def skew_symmetric_matrix_2_vector(self, M):
        # M 是反对称矩阵
        return np.array([M[2, 1], M[0, 2], M[1, 0]])



if __name__ == "__main__":
    R = RAxisAngle(theta=(math.pi/2), vector=np.mat([0, 0, 1], dtype='int64')).thetavector_2_R()
    
    print(SO3.M)
    print(SO3.so3)











