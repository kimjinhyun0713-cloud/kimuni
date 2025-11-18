from setuptools import setup, find_packages

setup(
    name='kimuni',
    version='0.0.0',
    description="description", 
    author='KIM JINHYUN', 
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'kimuni': ['./misc/**/*', './examples/**/*'],
    },
)
