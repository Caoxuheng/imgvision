import setuptools

with open('README.md','r') as fh:
    description = fh.read()

setuptools.setup(
    name='imgvision',
    version = '0.0.5',
    author = 'Xuheng Cao',
    author_email = 'caoxuhengcn@gmail.com',
    description = 'A package for image vision',
    long_description = 'An image vision package. It may help you achieve spectral image visualization, convert the spectral image to many kinds of color space, such as sRGB, Adobe RGB, and CIE 1964XYZ.'
                       'Besides, it includes some image process operation, evaluation, downsample, cluster prediction, and so on  ',
    url = 'https://github.com/Caoxuheng/imgvision',
    classifiers = ['License :: OSI Approved :: MIT License',
                   ],
    inculde_package_data = True,
    install_requires = ['numpy'],
    packages =['imgvision'],

)