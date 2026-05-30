from setuptools import setup
import os
from glob import glob

package_name = 'ros2_vision_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models', 'botellas_vs_latas_ncnn'), 
         glob('../../../IA/models/botellas_vs_latas_ncnn/*.bin') + 
         glob('../../../IA/models/botellas_vs_latas_ncnn/*.param') + 
         glob('../../../IA/models/botellas_vs_latas_ncnn/*.yaml') + 
         glob('../../../IA/models/botellas_vs_latas_ncnn/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Paquete de vision con IA',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nodo_vision = ros2_vision_pkg.nodo_vision:main',
        ],
    },
)
