import numpy as np
import os

path_ =os.path.dirname(os.path.realpath(__file__))

def downsample(img,scale):
    # img:被下采样图像
    # scale:下采样因子，如2，即原图像长宽各缩小2倍
    # 最后更新：2021年6月25日 曹栩珩
    info = np.shape(img)
    r = range(0, info[0], scale)
    l = range(0, info[1], scale)
    t_r = range(info[0])
    t_l =range(info[1])
    r = list(set(t_r)-set(r))
    l = list(set(t_l)-set(l))

    img = np.delete(img,r , axis=0)
    img = np.delete(img,l, axis=1)
    return img

def square_(n):
    import math
    i = int(math.sqrt(n))
    j = int(n/i)
    erro = n- i*j
    ind = 0

    while erro%i !=0 and erro%j !=0 :
        if ind%2 ==0:
            i-=1
        else:
            j-=1
        erro = n- i*j
        ind+=1

    if erro%i ==0:
        j+= erro/i
    else:
        i+= erro/j
    return (int(i),int(j))

class spectra():
    def __init__(self,illu='D65',illuminate = None) :
        # illu      :选择光源，D65，A，D50均为CIE规定光源信息；后缀L为实验室灯箱光源
        # illuminate:自行添加光源，在illu为'lab'时，可自行添加光源，需要shape为[31,]
        # self.S_xyz : CIE1964 10deg 下特定光源的刺激值函数。
        # 最后更新：2021年6月25日 曹栩珩
        self.illu = illu


        path = path_+ '/data/'
        if illu.upper() =='D65':
            self.S_xyz = np.load(path+'S_xyz_10.npy')
        elif illu.upper() == 'A':
            self.S_xyz = np.load(path+'S_xyz_A.npy')
        elif illu.upper() == 'D65L':
            self.S_xyz = np.load(path+'Lab_D65.npy')*0.5
            # print('65:\n', str(self.S_xyz).replace(' ', '\t').replace('[[', '').replace('[', ' ').replace(']', ''))
        elif illu.upper() == 'D75L':
            source = np.load(path+'A_D75.npy')*5
            light = np.empty([31, 3])
            cie64 = np.load(path+'cie64.npy')
            for i in range(3):
                light[:, i] = source[:,1]
            self.S_xyz = cie64 * light
        elif illu.upper() == 'AL':
            source = np.load(path + 'A_D75.npy')*5
            light = np.empty([31, 3])
            cie64 = np.load(path+'cie64.npy')
            for i in range(3):
                light[:, i] = source[:, 0]
            self.S_xyz = cie64 * light
            # print('A:\n',str(self.S_xyz).replace(' ','\t').replace('[[','').replace('[',' ').replace(']',''))
        elif illu.upper() == 'D50':
            self.S_xyz = np.load(path+'S_xyz_50.npy')

    def k(self):
        k = 100 / sum(self.S_xyz[:, 1])
        return k

    def space(self,img, space = 'adrgb',wp=None,xyz=False):
        # 将传入的高光谱图像转换至色空间
        # adrgb--Adobe RGB空间
        # srgb     -- sRGB空间
        # xyz       -- XYZ空间
        # gamma 对最后的输出rgb图进行gamma变换
        # 最后更新：2021年6月30日 曹栩珩
        self.info = np.shape(img)
        if xyz==True:
            self.XYZ = img
        else:
            self.rad_rs =  img.reshape(self.info[0] * self.info[1],self.info[2], 1)
            k = 100 / sum(self.S_xyz[:, 1])
            self.XYZ = np.sum(k * self.rad_rs * self.S_xyz, axis=1).reshape(self.info[0], self.info[1], 3)
        if space.lower() == 'xyz':
            return self.XYZ
        elif space.lower()=='adrgb':
            xyz = self.XYZ.T.reshape(3, self.info[0] * self.info[1])
            xyz = xyz / 100
            rgbtran = np.array([[2.04159, -0.56501, -0.34473], [-0.96924, 1.87597, 0.04156], [0.01344, -0.11836, 1.01517]])
            var_rgb = np.dot(rgbtran, xyz).reshape(3, self.info[1], self.info[0]).T
            var_rgb = pow(var_rgb,(1/2.19921875))
            return var_rgb
        elif space.lower()=='srgb':
            if wp:
                if self.illu.lower() == 'a' or self.illu.lower() == 'al':
                    M = np.array([[0.8652435, 0.0000000, 0.0000000], [0.0000000, 1.0000000, 0.0000000],
                                  [0.0000000, 0.0000000, 3.0598005]])
                    self.XYZ = np.dot(self.XYZ, M)
            xyz = self.XYZ.T.reshape(3, self.info[0] * self.info[1])

            xyz = xyz / 100
            rgbtran = np.array([[3.2404541, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])
            var_rgb = np.dot(rgbtran, xyz).reshape(3, self.info[1], self.info[0]).T
            for i in range(3):
                for j in range(self.info[0]):
                    for h in range(self.info[1]):
                        if var_rgb[j, h, i] > 0.0031308:
                            var_rgb[j, h, i] = 1.055 * (var_rgb[j, h, i] ** (1 / 2.4)) - 0.055
                        else:
                            var_rgb[j, h, i] = 12.92 * var_rgb[j, h, i]
            return var_rgb
        else:
            print('Erro:no such color space in the function')

    def sxyz(self):
        return self.S_xyz

    def space_spectrum(self,img, space = 'adrgb',wp=None,xyz = False):
        # 将传入的高光谱图像转换至色空间
        # adrgb--Adobe RGB空间
        # srgb     -- sRGB空间
        # xyz       -- XYZ空间
        # gamma 对最后的输出rgb图进行gamma变换
        # 最后更新：2021年6月30日 曹栩珩
        img = img.ravel()
        self.info = [1,np.shape(img)[0]]
        if xyz == True:
            self.XYZ = img
        else:
            self.rad_rs =  img.reshape(self.info[0] , self.info[1], 1)


            k = 100 / sum(self.S_xyz[:, 1])

            self.XYZ = np.sum(k * self.rad_rs * self.S_xyz, axis=1).reshape(self.info[0],  3)

        if space.lower() == 'xyz':
            return self.XYZ
        elif space.lower()=='adrgb':
            xyz = self.XYZ.T.reshape(3, self.info[0] )
            xyz = xyz / 100
            rgbtran = np.array([[2.04159, -0.56501, -0.34473], [-0.96924, 1.87597, 0.04156], [0.01344, -0.11836, 1.01517]])
            var_rgb = np.dot(rgbtran, xyz).reshape(3,  self.info[0]).T
            var_rgb = pow(var_rgb,(1/2.19921875))
            return var_rgb
        elif space.lower()=='srgb':
            if wp:
                if self.illu.lower() == 'a' or self.illu.lower() == 'al':
                    M = np.array([[0.8652435, 0.0000000, 0.0000000], [0.0000000, 1.0000000, 0.0000000],
                                  [0.0000000, 0.0000000, 3.0598005]])
                    self.XYZ = np.dot(self.XYZ, M)
            xyz = self.XYZ.T.reshape(3, self.info[0])

            xyz = xyz / 100
            rgbtran = np.array([[3.2404541, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])
            var_rgb = np.dot(rgbtran, xyz).reshape(3,  self.info[0]).T
            for i in range(3):
                for j in range(self.info[0]):
                        if var_rgb[j,  i] > 0.0031308:
                            var_rgb[j,  i] = 1.055 * (var_rgb[j, i] ** (1 / 2.4)) - 0.055
                        else:
                            var_rgb[j, i] = 12.92 * var_rgb[j, i]
            return var_rgb
        else:
            print('Erro:no such color space in the function')

class space():

    def rgb2hsv(img):
        img = img / 1
        c, l, s = img.shape

        min_m = np.min(img, axis=2)
        V = np.max(img, axis=2)
        S_ = (V - min_m)
        S = S_ / V
        HSV = np.empty([c, l, 3])
        HSV[:, :, 2] = V
        HSV[:, :, 1] = S

        for i in range(c):
            for j in range(l):
                if V[i, j] == 0:
                    HSV[i, j, 1] = 0
                if S[i, j] == 0:
                    HSV[i, j, 0] = 0
                else:
                    if V[i, j] == img[i, j, 0]:
                        HSV[i, j, 0] = 60 * (img[i, j, 1] - img[i, j, 2]) / S_[i, j]

                    elif V[i, j] == img[i, j, 1]:
                        HSV[i, j, 0] = 120 + 60 * (img[i, j, 1] - img[i, j, 0]) / S_[i, j]

                    elif V[i, j] == img[i, j, 2]:

                        HSV[i, j, 0] = 240 + 60 * (img[i, j, 0] - img[i, j, 1]) / S_[i, j]

                    if HSV[i, j, 0] < 0:
                        HSV[i, j, 0] += 360

        return HSV

    def hsv2rgb(img_):
        img = img_.copy()
        c, l, s = img.shape
        img[:, :, 0] /= 60
        RGB = np.empty([c, l, s])
        for i in range(c):
            for j in range(l):
                if img[i, j, 1] == 0:
                    RGB[i, j] = img[i, j, 2]
                else:
                    k = int(img[i, j, 0])
                    f = img[i, j, 0] - k
                    a = img[i, j, 2] * (1 - img[i, j, 1])
                    b = img[i, j, 2] * (1 - img[i, j, 1] * f)
                    c = img[i, j, 2] * (1 - img[i, j, 1] * (1 - f))
                    if k == 0:
                        RGB[i, j] = [img[i, j, 2], c, a]
                    elif k == 1:
                        RGB[i, j] = [b, img[i, j, 2], a]
                    elif k == 2:
                        RGB[i, j] = [a, img[i, j, 2], c]
                    elif k == 3:
                        RGB[i, j] = [a, b, img[i, j, 2]]
                    elif k == 4:
                        RGB[i, j] = [c, a, img[i, j, 2]]
                    elif k == 5:
                        RGB[i, j] = [img[i, j, 2], a, b]
        return RGB

    def srgb2xyz(img_):
        info = img_.shape
        img = img_.reshape([-1, 3])
        img[img <= 0.04045] /= 12.92
        img[img > 0.04045] = np.power((img[img > 0.04045] + 0.055) / 1.055, 2.4)
        trans = np.array([[0.4124564, 0.3575761, 0.1804357],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]])
        xyz = np.dot(trans, img.T).T.reshape(info)
        return xyz * 100

class distance():

    def cosine(a, b):
        assert a.shape == b.shape
        info = a.shape
        if len(info) == 3:
            a_ = a.reshape([-1, info[-1]])
            b_ = b.reshape([-1, info[-1]])
        else:
            a_ = a.copy()
            b_ = b.copy()

        cos_dist = np.empty(a_.shape[0])

        for i in range(a_.shape[0]):
            cos_dist[i] = np.dot(a_[i], b_[i].T) / (np.sqrt(np.dot(a_[i], a_[i].T) * np.dot(b_[i], b_[i].T)))

        return 1 - cos_dist

class cluster():

    def cosine_predict(img, centre):
        centre = centre.squeeze()
        img_ = img.reshape([-1, 3])
        img_ = np.tile(img_, (centre.shape[0], 1, 1))
        img_ = np.transpose(img_, (1, 0, 2))

        centre_mat = np.tile(centre, (img_.shape[0], 1, 1))

        dist = np.sum(centre_mat * img_, axis=-1) / np.sqrt(
            np.sum(centre_mat * centre_mat, axis=-1) * (np.sum(img_ * img_, axis=-1)))

        min_id = np.argmin(dist, axis=-1)
        return min_id