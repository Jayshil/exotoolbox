from setuptools import setup

setup(name='exotoolbox',
      version='0.1.1',
      description='The Exoplaneteers Toolbox',
      url='http://github.com/nespinoza/exotoolbox',
      author='Nestor Espinoza',
      author_email='nsespino@uc.cl',
      license='MIT',
      packages=['exotoolbox'],
      install_requires=['batman-package','emcee','lmfit','astropy','corner'],
      zip_safe=False)
