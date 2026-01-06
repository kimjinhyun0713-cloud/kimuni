from setuptools import setup, find_packages

setup(
    name='kimuni',
    version='0.1.0',
    description="description", 
    author='KIM JINHYUN', 
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'kimuni': ['./misc/**/*', './tools/**/*', 'cshmd/data/*']
    },
)
