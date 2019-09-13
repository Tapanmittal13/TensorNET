from setuptools import find_packages, setup

setup(name='Tensornet',
      version='0.0.1',
      description='API layer built on tensorflow-2.0 for high performance and easy training',
      url='https://github.com/Tapanmittal13/TensorNET.git',
      author='Tapan Mittal',
      author_email='tapanmittal13@gmail.com',
      license='MIT',
      install_requires=[
          'numpy','pandas','fastnumbers','more-itertools',
          'dill','seaborn','joblib','opencv-python',
      ],
      keywords=['tensorflow','neural networks','data-science','deep learning', 'machine learning','computer vision'],
      packages=find_packages(),
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)