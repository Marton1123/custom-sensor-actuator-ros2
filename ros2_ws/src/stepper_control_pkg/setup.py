from setuptools import find_packages, setup
import os
from glob import glob

PACKAGE_NAME = "stepper_control_pkg"

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # Registro ament (obligatorio)
        ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE_NAME]),
        # package.xml (obligatorio)
        ("share/" + PACKAGE_NAME, ["package.xml"]),
        # Archivos launch — *.py captura kiosco_launch.py y cualquier futuro launch
        (os.path.join("share", PACKAGE_NAME, "launch"), glob("launch/*.py")),
        # Archivos de configuración YAML (si los hay)
        (os.path.join("share", PACKAGE_NAME, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Maintainer Name",
    maintainer_email="maintainer@example.com",
    description="Paquete de control de motor paso a paso NEMA 17 con TB6600 en Raspberry Pi 5.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Formato:  "nombre_ejecutable = paquete.modulo:funcion_main"
            "nodo_sensores   = stepper_control_pkg.nodo_sensores:main",
            "nodo_actuadores = stepper_control_pkg.nodo_actuadores:main",
            "nodo_camara     = stepper_control_pkg.nodo_camara:main",
            "nodo_gui        = stepper_control_pkg.nodo_gui:main",
        ],
    },
)
