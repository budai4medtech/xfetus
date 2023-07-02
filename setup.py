from setuptools import setup, find_packages

setup(
  name = 'medisynth',
  packages = find_packages(exclude=['data', 'dependencies', 'event', 'examples', 'references', 'slides']),
  version = '0.0.0',
  license='MIT',
  description = 'Ultrasound Fetal Brain Imaging Synthesis',
  author = 'Harvey Mannering and Miguel Xochicale',
  author_email = '',
  url = 'https://github.com/mxochicale/medisynth/',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'diffusion models'
  ],
  install_requires=[
    'torch',
  ],
  classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8',
  ],
)