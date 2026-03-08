# import required libraries
from setuptools import find_packages
from setuptools import setup

# open requirements.txt with context manager
with open('requirements.txt') as f:
    # load project_lib_requirements.txt context as a list
    content = f.readlines()

# set requirements as rows that don't have git+
project_lib_requirements = [x.strip() for x in content if 'git+' not in x]

# difine project setup
setup(name='project_lib', # project name
      version="0.1", # project version
      description="Job Application Case", # project description
      packages=find_packages(), # find package from source folder
      install_requires=project_lib_requirements, # install packages in requirements list
      test_suite='tests', # folder with tests
      # include_package_data: to install data from MANIFEST.in
      include_package_data=False, # include data package inside package
    #   scripts=['scripts/project_lib_run'], # available scripts
      zip_safe=False) # project canNOT be installed and run from a zip file