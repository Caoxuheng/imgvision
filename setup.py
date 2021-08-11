import setuptools

with open('README.md','r') as fh:
    description = fh.read()

setuptools.setup(
    name='imgvision',
    version = '0.0.2',
    author = 'Xuheng Cao',
    author_email = 'caoxuhengcn@gmail.com',
    description = 'A package for image vision',
    long_description = description ,
    url = 'https://github.com/Caoxuheng/imgvision',
    classifiers = ['License :: OSI Approved :: MIT License',
                   ],
    inculde_package_data = True,
    install_requires = ['numpy'],
    packages =['imgvision'],

)