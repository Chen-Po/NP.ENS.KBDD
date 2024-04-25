import setuptools


setuptools.setup(
    name='KBDD',
    version='0.1.5',
    packages=setuptools.find_packages(),
    description='A package for KBDD: a knowledge-based and data-driven method for genetic network construction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Chen-Po Liao',
    author_email='liaochenpo@gmail.com',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    python_requires='>=3.6',
)
