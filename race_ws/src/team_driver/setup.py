import os
from glob import glob

from setuptools import setup

package_name = 'team_driver'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, package_name + '.core', package_name + '.tools'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Team',
    maintainer_email='you@e.ntu.edu.sg',
    description='Hackathon entry.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Keep this entry: the judges run `ros2 run team_driver driver`.
            # Add more of your own alongside it if you like.
            'driver = team_driver.driver:main',
            # Development tools. Not part of a scored run.
            'sysid_drive = team_driver.tools.sysid_drive:main',
            'map_record = team_driver.tools.map_record:main',
        ],
    },
)
