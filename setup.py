from setuptools import setup, find_packages

setup(
    name="mantis",
    version="2.0.0",
    author="Carson Dudley",
    description="Simulation-grounded foundation model for infectious disease forecasting",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
    ],
    extras_require={
        "v2": ["chronos-forecasting==2.3.2"],
    },
    include_package_data=True,
    python_requires=">=3.7",
)
