import setuptools

with open('README.md','r') as fh:
    description = fh.read()

setuptools.setup(
    name='imgvision',
    version = '0.0.1',
    author = 'Xuheng Cao',
    author_email = 'caoxuhengcn@gmail.com',
    description = 'A package for image vision',
    long_description = description ,
    url = '',
    install_requires = ['numpy'],
    packages = setuptools.find_packages(),

)