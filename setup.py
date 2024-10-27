import setuptools


setuptools.setup(
    name='NEKBDD',
    version='0.1.0',
    packages=setuptools.find_packages(),
    description='A package for NEKBDD: a nonparametric ensemble knowledge-based and data-driven method for genetic network construction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Chen-Po Liao',
    author_email='liaochenpo@gmail.com',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    install_requires=[
        'numpy',
        'pandas',
        'powerlaw'
    ],
    python_requires='>=3.6',
)
