from setuptools import setup
import setuptools
setup(name='mylib',
      version='0.1',
      description='1D Heat Transport Models',
      author='Robin Keegan-Treloar',
      author_email='robin_kt@outlook.com',
      url='https://github.com/robinkeegan/mylib',
      license='MIT',
      packages=setuptools.find_packages(),
      zip_safe=False,
      install_requires=[
            "pandas >= 0.23.1",
            "numpy >= 1.14.5",
            "scipy >= 1.1.0",
            "statsmodels >= 0.9.0",
      ])
