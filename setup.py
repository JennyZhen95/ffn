from setuptools import setup
# entry_points = {
#   'console_scripts':[
#      'n4n_run = n4n.n4n_run:main'
#   ]
# }
entry_points = {}

setup(name='ffn',
      packages=['ffn'],
      entry_points=entry_points,
      include_package_data=True,
      version='1.0.0',
      install_requires = [],
)
