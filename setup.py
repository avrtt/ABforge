from setuptools import find_packages, setup

setup(
    name="abforge",
    packages=find_packages(include=["abforge", "abforge.stats", "abforge.utils"]),
    version="2.12.3",
    description="A feature-rich library for running Bayesian A/B tests",
    author="Vladislav Averett",
    license="",
    install_requires=[
        "numpy==1.26.4",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest==4.4.1"],
    test_suite="tests",
)
