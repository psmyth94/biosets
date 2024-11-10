from setuptools import find_packages, setup

REQUIRED_PKGS = [
    "datasets>=2.19.1",
    "biocore",
]

QUALITY_REQUIRE = ["ruff>=0.1.5"]

DOCS_REQUIRE = [
    # Might need to add doc-builder and some specific deps in the future
    "s3fs",
]

TESTS_REQUIRE = [
    "pytest",
    "pytest-timeout",
    "pytest-xdist",
    "scipy",
    "polars>=0.20.5",
    "timezones>=0.10.2",
]

EXTRAS_REQUIRE = {
    "polars": ["polars>=0.20.5", "timezones>=0.10.2"],
    "apache-beam": ["apache-beam>=2.26.0,<2.44.0"],
    "vcf": ["cyvcf2>=0.30.0", "sgkit>=0.0.1"],
    "tensorflow": [
        "tensorflow>=2.2.0,!=2.6.0,!=2.6.1; sys_platform != 'darwin' or platform_machine != 'arm64'",
        "tensorflow-macos; sys_platform == 'darwin' and platform_machine == 'arm64'",
    ],
    "tensorflow_gpu": ["tensorflow-gpu>=2.2.0,!=2.6.0,!=2.6.1"],
    "torch": ["torch"],
    "jax": ["jax>=0.3.14", "jaxlib>=0.3.14"],
    "s3": ["s3fs"],
    "scipy": ["scipy"],
    "test": QUALITY_REQUIRE + TESTS_REQUIRE + DOCS_REQUIRE,
    "quality": QUALITY_REQUIRE,
    "docs": DOCS_REQUIRE,
}

setup(
    name="biosets",
    version="1.2.0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="Bioinformatics datasets and tools",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Patrick Smyth",
    author_email="psmyth1994@gmail.com",
    url="https://github.com/psmyth94/biosets",
    download_url="https://github.com/psmyth94/biosets/tags",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    python_requires=">=3.8.0,<3.12.0",
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="omics machine learning bioinformatics datasets",
    zip_safe=False,  # Required for mypy to find the py.typed file
)
