try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Pegasos: an sklearn-like package for fitting pegasos SVMs',
    'author': 'Eddie Bell',
    'url': 'github.com/ejlb/pegasos',
    'author_email': 'eddie@lyst.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['pegasos'],
    'scripts': [],
    'name': 'pegasos',
    'install_requires': 'scikit-learn>=0.13',
}

setup(**config)
