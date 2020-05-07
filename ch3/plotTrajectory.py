import numpy as np
from tools3d import Quateration, world_coordinate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def convert_all_points(quater_t, quater_q, quater_p):
    # 把每一个四元数形式转换到世界坐标
    X = []
    Y = []
    Z = []
    for i in range(len(quater_t)):
        q2 = world_coordinate(q1=quater_q[i], t1=quater_t[i], p1=quater_p[i])
        X.append(q2.x)
        Y.append(q2.y)
        Z.append(q2.z)
    return X,Y,Z

def plot_point_cloud(X,Y,Z):
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X,Y,Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


if __name__ == "__main__":
    trajectory_file = './trajectory.txt'
    time = []
    quater_t = []  # 平移
    quater_q = []  # 世界坐标到机器人坐标的选择四元数
    quater_p = []  # 机器人坐标系下的坐标

    with open(trajectory_file, 'r') as f:
        for line in f:
            line = line.split(' ')
            time.append(float(line[0]))
            quater_q.append(Quateration(s=float(line[0]),x=float(line[0]),y=float(line[0]),z=float(line[0])))
            quater_t.append(Quateration(s=0,x=float(line[1]),y=float(line[2]),z=float(line[3])))
            quater_p.append(Quateration(s=0,x=0,y=0,z=0))

    X,Y,Z = convert_all_points(quater_t, quater_q, quater_p)
    plot_point_cloud(X, Y, Z)




# reference: https://www.cnblogs.com/wuwen19940508/p/8638266.html

