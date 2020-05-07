import numpy as np

class Quateration():
    def __init__(self, s, x, y, z):
        self.s = s
        self.x = x
        self.y = y
        self.z = z

    def add(self, quater2):
        # 加法微信
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
        return Quateration(s=self.s, x=-1*self.x, y=-1*self.y, z=-1*self.z)

    def inverse(self):
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


    def print_self(self):
        print("the quater is: {} + {}*i + {}*j + {}*k".format(self.s, self.x, self.y, self.z))


def world_coordinate(q1, t1, p1):
    # q 是旋转四元数 t是平移 p是机器人坐标
    world = q1.inverse().dot(p1.sub(t1)).dot(q1)
    return world


if __name__ == "__main__":
    q1 = Quateration(s=0.35,x=0.2,y=0.3,z=0.1)
    q2 = Quateration(s=-0.5,x=0.4,y=-0.1,z=0.2)

    t1 = Quateration(s=0,x=0.3,y=0.1,z=0.1)
    t2 = Quateration(s=0,x=-0.1,y=0.5,z=0.3)

    p1 = Quateration(s=0,x=0.5,y=0,z=0.2)

    # 设在世界坐标下某点坐标为 tmp
    # p1 = q1 * tmp + t1, p2 = q2 * tmp + t2
    # => tmp = q1^(-1) * (p1-t1)  p2 = q2 * tmp + t2
    #tmp = q1.inverse().dot(p1.sub(t1)).dot(q1)
    #p2 = q2.dot(tmp).dot(q2.inverse()).add(t2)
    #p2.print_self()
    world_coordinate(q1, t1, p1).print_self()
    ########### 注意，四元数坐标变换公式是: p2 = q * p1 * q^-1








