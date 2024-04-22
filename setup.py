import setuptools

with open('README.md','r') as fh:
    description = fh.read()

setuptools.setup(
    name='imgvision',
    version = '0.1.7.2',
    author = 'Xuheng Cao',
    author_email = 'caoxuhengcn@gmail.com',
    description = 'A package for image vision',
    long_description = 'An image vision package. It helps you achieve spectral image visualization, convert the spectral image to different color spaces, such as sRGB, Adobe RGB, CIE XYZ, CIE Lab, and etc.'
                       'Besides, it includes some image process operation, evaluation, downsample, cluster prediction, and image evaluation metrics ',
    url = 'https://blog.csdn.net/syuhen/category_11306284.html',
    classifiers = ['License :: OSI Approved :: MIT License',
                   ],
    package_data = {

            '': ['*.npy'],
            # # 包含demo包data文件夹中的 *.dat文件
            # 'demo': ['data/*.dat'],
        },
    inculde_package_data = True,
    install_requires = ['numpy'],
    packages =setuptools.find_packages(),
)
