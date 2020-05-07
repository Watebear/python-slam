import numpy as np

def normalize_vector(vector):
    # 归一化得到单位长度的方向向量
    sum_square = np.sum(np.square(vector))
    if sum_square != 0:
        vector = vector / sum_square
    else :
        assert "the sum square equals 0, can't normalized by vector/sum_square"
    return vector

def skew_symmetric_matrix(vector):
    # 反对称矩阵 a^
    return np.array([[0, -1*vector[0,2], vector[0,1]],
                     [vector[0,2], 0, -1*vector[0,0]],
                     [-1*vector[0,1], vector[0,0], 0]])

class Rotation():
    def __init__(self, R=None, t=None, theta=None, vector=None):
        # 验证旋转矩阵R是一个行列式为1的正交矩阵 !!!
        self.R = R
        self.t = t
        self.theta = theta
        self.vector = vector
        #self.T = np.hstack(np.vstack(R,[0]), np.vstack(t.T,1))

    def check(self):
        # 保证旋转矩阵和旋转轴之间至少有一种合法输入
        if (self.R is None) and (self.theta is None and self.vector is None):
            assert "init error, check your input"
        # 如果同时用两种方式初始化，则要保证没有冲突
        # 验证R是政教矩阵

    def R_2_thetavector(self):
        theta = np.arccos((self.R.trace()-1)/2)
        eigvalue, eigvector = np.linalg.eig(self.R)
        print('eigenvalue:', eigvalue)
        print("eigenvector", eigvector)

        index = np.argwhere(eigvalue==1).reshape([1,])
        print('index:', index)
        if len(index) != 1:
            index = index[0]
        e_vector = eigvector[index]
        e_vector = normalize_vector(e_vector).reshape([-1])

        return theta, e_vector

    def thetavector_2_R(self):
        # 罗德里格斯公式： R = COS@I + (1-cos@)nn^T + sin@ n^
        R = np.cos(self.theta)*np.eye(3) + \
            (1-np.cos(self.theta))*(self.vector.T.dot(self.vector)) + \
            np.sin(self.theta)*skew_symmetric_matrix(self.vector)
        return R

    '''
    验证代码：
    rot1 = Rotation(theta=3.14, vector=np.mat([0,0,1]))
    R1 = rot1.thetavector_2_R()
    rot2 = Rotation(R=R1)
    theta, vector = rot2.R_2_thetavector()
    print(theta, vector)
    '''





if __name__ == "__main__":







