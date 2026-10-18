from setuptools import setup, find_packages

# Maps the folder name (light_mem) to the expected package name (lightmem).
# Run: pip install -e baselines/light_mem/
setup(
    name="lightmem",
    version="0.1.0",
    package_dir={"lightmem": "."},
    packages=["lightmem"] + [
        "lightmem." + pkg
        for pkg in find_packages(".")
    ],
)
