from setuptools import setup, find_packages

setup(
    name='mushr_mujoco_gym',
    version='0.0.1',
    packages=find_packages(),
    package_data={
        'mushr_mujoco_gym': ['include/models/*.xml', 'include/meshes/*.stl'],
    },
    include_package_data=True,
    install_requires=['gym>=0.2.3']
)
