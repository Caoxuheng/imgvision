import numpy as np
import os
import warnings

path_ = os.path.dirname(os.path.realpath(__file__))


class imgshape():
    def __init__(self, img):
        self.info = img.shape
        self.img = img

    def flatten(self):
        return self.img.reshape([-1, self.info[-1]])

    def inv_flatten(self, img2):
        return img2.reshape(list(self.info[:2]) + [-1])

def add_noise(img,SNR):
    '''

    :param img:待添加噪音图像
    :param SNR: 噪音信噪比
    :return: 加噪音后的图像
    最后更新 2022年6月16日 曹栩珩
    '''
    M,N,B = img.shape
    gamma = np.sqrt(pow(img, 2).sum() / (10 ** (SNR / 10)) / (img.size))
    img += gamma*np.random.randn(M,N,B)
    return img


def get_blur(type):
    if type =='motion':
        kernel = np.load(path_+'/data/blur/kernel_m19.npy')
        k1,k2 = kernel.shape
    return kernel.reshape([1,1,k1,k2])

def block_extract(img, row, line, inc, r):
    value = []
    scale = int(r / 3)
    c, l, s = img.shape
    for i in range(row):
        for j in range(line):
            x, y = r + i * (inc + 2 * r), j * (inc + 2 * r) + r

            box = img[x - scale:x + scale, y - scale:y + scale]
            value.append(np.mean(box.reshape([-1, s]), axis=0))
    return np.asarray(value)

def downsample(img, scale, type='spatial', band2=np.array(range(400, 710, 10)),Band = np.arange(400,710,10)):
    '''
        :img 被下采样图像
        :scale 下采样因子，如2，即原图像长宽各缩小2倍
        :type spatial/spectral 可选 分别表示空间和光谱下采样
        最后更新：2022年6月16日 曹栩珩
    '''

    if type.lower() == 'spatial':
        img = img [scale//2-1::scale,scale//2-1::scale]
    elif type.lower() == 'spectral_iq':
        '''
        Specim-IQ 专用； 由397~780 超至 任意维度
        需要传入band2 列表，进行插值
        '''
        Band = np.array(
            [397.32, 400.2, 403.09, 405.97, 409, 412, 415, 418, 420, 423, 426, 429, 432, 435, 438, 441, 443, 446, 449,
             452, 455, 458,
             461, 464, 467, 469, 472, 476, 478, 481, 484, 487, 490, 493, 496, 499, 502, 505, 508, 510, 513, 516, 519,
             522, 525, 528, 531, 534, 537, 540, 543, 545, 548, 551, 554, 557, 560, 563, 566, 569, 572, 575, 577, 581,
             584, 587, 590, 593, 596, 599, 602, 605, 607, 610, 613, 616, 619, 622, 625, 628, 631, 634, 637, 640, 643,
             646, 649, 652, 655, 658, 661, 664, 666, 669, 672, 676, 679, 682, 685, 688, 691, 694, 697, 700, 702.58,
             705.57, 708.57, 711.56, 714.55, 717.54, 720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53,
             744.53, 747.54, 750.54,
             753.55, 756.56, 759.56, 762.57, 765.58, 768.6, 771.61, 774.62, 777.64, 780.65])

        _img = imgshape(img)
        img = _img.flatten()

        img_ = np.empty([img.shape[0]] + list(band2.shape))
        for x in range(img.shape[0]):
            img_[x] = np.interp(band2, Band, img[x])

        img = _img.inv_flatten(img_)

    elif type.lower() == 'spectral':
        '''
        对光谱图像进行光谱维度上采样
        '''

        _img = imgshape(img)
        img = _img.flatten()

        img_ = np.empty([img.shape[0]] + list(band2.shape))
        for x in range(img.shape[0]):
            img_[x] = np.interp(band2, Band, img[x])

        img = _img.inv_flatten(img_)

    return img

def square_(n):
    import math
    i = int(math.sqrt(n))
    j = int(n / i)
    erro = n - i * j
    ind = 0

    while erro % i != 0 and erro % j != 0:
        if ind % 2 == 0:
            i -= 1
        else:
            j -= 1
        erro = n - i * j
        ind += 1

    if erro % i == 0:
        j += erro / i
    else:
        i += erro / j
    return (int(i), int(j))

class cvtcolor():

    def lab2xyz(self, lab,illuminant='D65'):
        s_xyz = spectra(illuminant).sxyz()
        w = np.sum(s_xyz,axis=0).reshape([1,1,3])
        w/=w[:,:,1]
        T = 4 / 29
        __xyz = np.empty(lab.shape)
        __xyz[:, :, 1] = (1 / 116) * (lab[:, :, 0] + 16)
        __xyz[:, :, 0] = __xyz[:, :, 1] + (1 / 500) * lab[:, :, 1]
        __xyz[:, :, 2] = __xyz[:, :, 1] - (1 / 200) * lab[:, :, 2]
        fxyz = np.empty(lab.shape)
        fxyz[__xyz > T] = __xyz[__xyz > T] ** 3
        fxyz[__xyz <= T] = 3 * T ** 2 * (__xyz[__xyz <= T] - T)
        return fxyz * w

    def lab2rgb(self,lab,space = 'srgb',illuminant ='D65'):
        xyz = self.lab2xyz(lab,illuminant=illuminant)
        if space.lower() =='srgb':
            return self.xyz2srgb(xyz)
        elif space.lower() =='adobergb':
            return  self.xyz2adrgb(xyz)

    def rgb2xyz(self,img_,type ='srgb',wp = 'd65'):

        info = img_.shape
        img = img_.reshape([-1, 3])
        if type.lower() =='srgb' or type.lower() =='apple' or type.lower() =='adp3' or type.lower() =='apple display p3' or type.lower() =='display p3':

            if type.lower() =='srgb':
                trans = np.array([[0.4124564, 0.3575761, 0.1804357],
                                  [0.2126729, 0.7151522, 0.0721750],
                                  [0.0193339, 0.1191920, 0.9503041]])
            elif type.lower() == 'apple' or type.lower() == 'adp3' or type.lower() == 'apple display p3' or type.lower() == 'display p3':
                trans = np.array([[0.4866, 0.2657, 0.1982],
                                  [0.2290, 0.6917, 0.0793],
                                  [0.0, 0.0451, 1.0437]])
            img[img <= 0.04045] /= 12.92
            img[img > 0.04045] = np.power((img[img > 0.04045] + 0.055) / 1.055, 2.4)

            xyz = np.dot(trans, img.T).T.reshape(info)

        elif type.lower() =='adobergb':
            img **=2.19921875
            trans = np.linalg.inv(np.array(
                [[2.04159, -0.56501, -0.34473], [-0.96924, 1.87597, 0.04156], [0.01344, -0.11836, 1.01517]]))

            xyz = np.dot(trans, img.T).T.reshape(info)
        else:
            xyz = 0

        s_xyz = spectra(wp).sxyz()
        w = np.sum(s_xyz,axis=0).reshape([1,1,3])
        D65_w = np.sum(spectra('d65').sxyz(),axis=0).reshape([1,1,3])
        xyz*=100
        xyz*=w/D65_w
        return xyz

    def xyz2lab(self,xyz,illuminant = 'D65'):
        s_xyz = spectra(illuminant).sxyz()
        w = np.sum(s_xyz, axis=0).reshape([1, 1, 3])
        w /= w[:, :, 1]
        T = pow(6/29,2)
        fxyz = np.empty(xyz.shape)
        __lab = np.empty(xyz.shape)
        fx = xyz[:, :, 0] / w[:, :, 0]
        fy = xyz[:, :, 1] / w[:, :, 1]
        fz = xyz[:, :, 2] / w[:, :, 2]
        fxyz[:,:,0][fx>T] = fx[fx > T] ** (1 / 3)
        fxyz[:, :, 0][fx <= T] =(1/3)*pow(29/6,2)* fx[fx <= T] +16/166
        fxyz[:,:,1][fy>T] = fy[fy > T] ** (1 / 3)
        fxyz[:, :, 1][fy <= T] =(1/3)*pow(29/6,2)* fy[fy <= T] +16/166
        fxyz[:,:,2][fz>T] = fz[fz > T] ** (1 / 3)
        fxyz[:, :, 2][fz <= T] =(1/3)*pow(29/6,2)* fz[fz <= T] +16/166

        __lab[:, :, 0] = 116 * fxyz[:,:,1] - 16
        __lab[:, :, 1] =  500 * (fxyz[:,:,0]-fxyz[:,:,1])
        __lab[:, :, 2] = 200 * (fxyz[:,:,1]-fxyz[:,:,2])
        return __lab

    def xyz2srgb(self,xyz):
        rgbtran = np.array([[3.2404541, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560],
                            [0.0556434, -0.2040259, 1.0572252]])
        var_rgb = np.dot( xyz,rgbtran.T)
        var_rgb_ = var_rgb.copy()
        var_rgb_[var_rgb > 0.0031308] = 1.055 * (var_rgb[var_rgb > 0.0031308] ** (1 / 2.4)) - 0.055
        var_rgb_[var_rgb <= 0.0031308] = 12.92 * var_rgb[var_rgb <= 0.0031308]
        return var_rgb_

    def xyz2adrgb(self,xyz):
        rgbtran = np.array([[2.04159, -0.56501, -0.34473], [-0.96924, 1.87597, 0.04156], [0.01344, -0.11836, 1.01517]])
        var_rgb = np.dot( xyz,rgbtran)
        return np.power(var_rgb,1 / 2.19921875)

    def rgb2hsv(self,img):
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

    def rgb2lab(self,rgb,type = 'srgb',illuminant = 'D65'):

        return self.xyz2lab(self.rgb2xyz(rgb,type,wp=illuminant)/100,illuminant)

    def hsv2rgb(self,img_):
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

class spectra():

    def __init__(self, illuminant='D65',band = np.arange(400,710,10),deg = 10,band2 = None):
        '''
        创建一个光谱转换器。该光谱转换器将需要预先定义光源类型，光源波段以及视场。
        注意：自定义的光源信息需要填充band信息，否则将自动压缩至400nm~700nm，10nm间隔。
        :param illuminant: 字符串或数组| 选择光源，D65，A，D50均为CIE规定光源信息.
        :param band: 数组 | 光谱波段，默认400nm~700nm,10nm间隔
        :param deg: 整数 | CIE 视角，默认10°，可选2°
        最后更新：2021年10月28日 曹栩珩
        '''
        path = path_ + '/data/'
        bar_xyz = np.load(path + 'CIExyz_deg'+str(deg)+'.npy')
        __bar_xyz = np.empty([band.shape[0],3])
        for i in range(3):
            __bar_xyz[:, i] = np.interp(band, np.arange(360, 805, 5), bar_xyz[:,i])

        if type(illuminant) == str:
            self.illu = illuminant.lower()
            illuminant_list = ['a','d65','d50','b','c','d55']
            if self.illu not in illuminant_list:
                print(f"Unkown illuminant{self.illu}, only support 'A','D65','D50', 'D55', 'B', 'C' or the custom")
            _illuminance = np.load(path + self.illu+'.npy')
            self.illuminance = np.interp(band,np.arange(360, 805, 5),_illuminance)
        else:
            if band2 is None:
                band2 = band
            self.illuminance =  np.interp(band,band2,illuminant)
        if len(self.illuminance.shape) ==1:
            self.illuminance = self.illuminance.reshape([-1,1])
        self.S_xyz = self.illuminance * __bar_xyz


    def k(self):
        k = 100 / sum(self.S_xyz[:, 1])
        return k



    def space(self, img, space='adrgb', wp=None, xyz=False):
        '''

        :param img: 传入待转换的高光谱图像或XYZ图像（XYZ图像需填写xyz=True）
        :param space: 字符串 | 转换的目标空间，默认AdobeRGB，可选’sRGB‘，’XYZ‘
        :param wp: 是否进行白点校正
        :param xyz: XYZ图像校正
        :return: 目标颜色空间图像
        最后更新：2021年6月30日 曹栩珩
        '''

        self.info = np.shape(img)

        if xyz == True:
            self.XYZ = img
        else:
            print(self.S_xyz.shape)
            k = 100 / sum(self.S_xyz[:, 1])
            self.XYZ = k*img @ self.S_xyz
        if space.lower() == 'xyz':
            return self.XYZ
        elif space.lower() == 'adrgb':
            xyz = self.XYZ.T.reshape(3, self.info[0] * self.info[1])
            xyz = xyz / 100
            rgbtran = np.array(
                [[2.04159, -0.56501, -0.34473], [-0.96924, 1.87597, 0.04156], [0.01344, -0.11836, 1.01517]])
            var_rgb = np.dot(rgbtran, xyz).reshape(3, self.info[1], self.info[0]).T
            var_rgb = pow(var_rgb, (1 / 2.19921875))
            return var_rgb
        elif space.lower() == 'srgb':
            if wp:
                if self.illu.lower() == 'a' or self.illu.lower() == 'al':
                    M = np.array([[0.8652435, 0.0000000, 0.0000000], [0.0000000, 1.0000000, 0.0000000],
                                  [0.0000000, 0.0000000, 3.0598005]])
                    self.XYZ = np.dot(self.XYZ, M)
            xyz = self.XYZ.T.reshape(3, self.info[0] * self.info[1])

            xyz = xyz / 100
            rgbtran = np.array([[3.2404541, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560],
                                [0.0556434, -0.2040259, 1.0572252]])
            var_rgb = np.dot(rgbtran, xyz).reshape(3, self.info[1], self.info[0]).T
            var_rgb_ = var_rgb.copy()
            var_rgb_[var_rgb>0.0031308] = 1.055 * (var_rgb[var_rgb>0.0031308] ** (1 / 2.4)) - 0.055
            var_rgb_[var_rgb <= 0.0031308] = 12.92*var_rgb[var_rgb <= 0.0031308]

            return var_rgb_
        elif space.lower() == 'nkd700':

            assert img.shape[-1]==31 ,f'NikonD700目前只能处理10nm为间隔，波长范围400nm~700nm的光谱图像，目前接收到的图像维度{img.shape}'
            return img @ np.load(path_+'/data/SRF/NikonD700.npy')
        else:
            warnings.warn('Not found such a color space. if you need it, please contact caoxuhengcn@gmail.com')

    def sxyz(self):
        return self.S_xyz

    def space_spectrum(self, img, space='adrgb', wp=None, xyz=False):

        img = img.ravel()
        self.info = [1, np.shape(img)[0]]
        if xyz == True:
            self.XYZ = img
        else:
            self.rad_rs = img.reshape(self.info[0], self.info[1], 1)

            k = 100 / sum(self.S_xyz[:, 1])

            self.XYZ = np.sum(k * self.rad_rs * self.S_xyz, axis=1).reshape(self.info[0], 3)

        if space.lower() == 'xyz':
            return self.XYZ
        elif space.lower() == 'adrgb':
            xyz = self.XYZ.T.reshape(3, self.info[0])
            xyz = xyz / 100
            rgbtran = np.array(
                [[2.04159, -0.56501, -0.34473], [-0.96924, 1.87597, 0.04156], [0.01344, -0.11836, 1.01517]])
            var_rgb = np.dot(rgbtran, xyz).reshape(3, self.info[0]).T
            var_rgb = pow(var_rgb, (1 / 2.19921875))
            return var_rgb
        elif space.lower() == 'srgb':
            if wp:
                if self.illu.lower() == 'a' or self.illu.lower() == 'al':
                    M = np.array([[0.8652435, 0.0000000, 0.0000000], [0.0000000, 1.0000000, 0.0000000],
                                  [0.0000000, 0.0000000, 3.0598005]])
                    self.XYZ = np.dot(self.XYZ, M)
            xyz = self.XYZ.T.reshape(3, self.info[0])

            xyz = xyz / 100
            rgbtran = np.array([[3.2404541, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560],
                                [0.0556434, -0.2040259, 1.0572252]])
            var_rgb = np.dot(rgbtran, xyz).reshape(3, self.info[0]).T
            for i in range(3):
                for j in range(self.info[0]):
                    if var_rgb[j, i] > 0.0031308:
                        var_rgb[j, i] = 1.055 * (var_rgb[j, i] ** (1 / 2.4)) - 0.055
                    else:
                        var_rgb[j, i] = 12.92 * var_rgb[j, i]
            return var_rgb
        else:
            raise AttributeError(f'{space}空间不在预定空间范围内，请联系caoxuhengcn@gmail.com')




class spectra_metric():
    '''
      光谱评价指标 待测图像 x1(shape:[m,n,b]),x2(shape:[m,n,b]);其中m,n为空间分辨率，b为波段数
    '''

    def __init__(self, x1, x2, max_v=1, scale=16):
        self.scale = scale
        self.info = x1.shape
        self.max_v = max_v
        if len(self.info) == 3:
            self.x1_ = x1.reshape([-1, self.info[-1]])
            self.x2_ = x2.reshape([-1, self.info[-1]])
        else:
            self.x1_ = x1
            self.x2_ = x2

    def SAM(self, mode=''):
        A = np.sum(self.x1_ * self.x2_, axis=-1) / (
                    np.sqrt(np.sum(self.x1_ * self.x1_, axis=-1)) * np.sqrt(np.sum(self.x2_ * self.x2_, axis=-1)))
        _SAM = np.arccos(A) * 180 / np.pi
        if mode == 'mat':
            return _SAM
        else:
            return np.mean(_SAM)

    def MSE(self, mode=''):
        if mode == 'mat':

            self.MSE_mat = np.mean(np.power(self.x1_ - self.x2_, 2), axis=0)

            return self.MSE_mat
        else:
            return np.mean(np.power(self.x1_ - self.x2_, 2))

    def ERGAS(self):
        k = 100 / self.scale
        return k * np.sqrt(np.mean(self.MSE('mat') / np.power(np.mean(self.x2_, axis=0), 2)))

    def PSNR(self, mode=''):
        _PSNR = 10 * np.log10(np.power(self.max_v, 2) / self.MSE('mat'))
        if mode == 'mat':
            return _PSNR
        else:
            return np.mean(_PSNR)

    def SSIM(self, k1=0.01, k2=0.03, mode=''):
        l = self.max_v
        u1 = np.mean(self.x1_, axis=0).reshape([1, -1])
        u2 = np.mean(self.x2_, axis=0).reshape([1, -1])
        Sig1 = np.std(self.x1_, axis=0).reshape([1, -1])
        Sig2 = np.std(self.x2_, axis=0).reshape([1, -1])
        sig12 = np.sum((self.x1_ - u1) * (self.x2_ - u2), axis=0) / (self.info[0] * self.info[1] - 1)
        c1, c2 = pow(k1 * l, 2), pow(k2 * l, 2)
        SSIM = (2 * u1 * u2 + c1) * (2 * sig12 + c2) / ((u1 ** 2 + u2 ** 2 + c1) * (Sig1 ** 2 + Sig2 ** 2 + c2))
        if mode == 'mat':
            return SSIM
        else:
            return np.mean(SSIM)

    def CC(self, mode=''):
        x1_mean = self.x1_.T - self.x1_.mean(-1).T
        x2_mean = self.x2_.T - self.x2_.mean(-1).T
        up = np.sum(x1_mean * x2_mean, axis=0)
        down = np.sqrt(np.power(x1_mean, 2).sum(0) * np.power(x2_mean, 2).sum(0))
        CC = up.T / down.T
        if mode == 'mat':
            return CC
        else:
            return np.mean(CC)

    def get_Evaluation(self,  k1=0.01, k2=0.03):
        return self.PSNR(),self.SAM(),self.ERGAS(),self.SSIM(k1=k1, k2=k2),self.MSE(),self.CC()

    def Evaluation(self,idx=0,k1=0.01,k2=0.03):
        PSNR,SAM,ERGAS,SSIM,MSE,CC = self.get_Evaluation(k1,k2)
        print(f'{idx}\t{PSNR}\t{SAM}\t{ERGAS}\t{SSIM}\t{np.sqrt(MSE)}\t{CC}')

class distance():

    def cosine(self,a, b):
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

    def DEab(self,truth,sample,type ='mean'):
        matrix =  np.sqrt(np.sum(np.power(sample-truth,2),axis=-1))
        if type =='mean':
            return np.mean(matrix)
        elif type == 'mat':
            return matrix

    def DEab_RGB_LAB(self,RGB,LAB,type='srgb',illuminant='d65',m_type = 'mean'):
        LAB_S = cvtcolor().rgb2lab(RGB,type,illuminant)

        return self.DEab(LAB,LAB_S,m_type)

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


