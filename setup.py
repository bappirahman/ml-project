from setuptools import find_packages, setup
from typing import List

def get_requirements(filepath:str)->List[str]:
  """
  Reads the requirements from the given file path and returns them as a list of strings.
  
  Args:
      filepath (str): The path to the file containing the requirements.
  
  Returns:
      List[str]: A list of strings representing the requirements.
  """
  requirements = []
  with open(filepath, 'r') as file:
    requirements = file.readlines()
    requirements = [req.replace('\n', '') for req in requirements]

    if '-e .' in requirements:
      requirements.remove('-e .')
    return requirements


setup(
  name='ml-project',
  version='0.0.1',
  description='A End to End ML Project',
  author='Bappi Rahman',
  author_email='cs.bappirahman@gmail.com',
  packages=find_packages(),
  install_requires=get_requirements('./requirements.txt')
)