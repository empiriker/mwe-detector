import setuptools

setuptools.setup(
    name="mwe_detector",
    version="0.0.1",
    description="A SpaCy MWE identification pipeline component",
    url="https://github.com/empiriker/mwe-detector",
    author="Till ÜF",
    author_email="empiriker@yahoo.de",
    license="CC-BY-SA-4.0",
    packages=["mwe_detector"],
    zip_safe=False,
    include_package_data=True,
    package_data={"mwe_detector": ["data/*.json"]},
    python_requires=">=3.10",
    install_requires=["numpy>=1.15.0", "spacy>=3.7.2", "srsly>=2.4.6", "ujson>=5.8.0"],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz#egg=en_core_web_sm",
            "fr_core_news_sm @ https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.7.0/fr_core_news_sm-3.7.0.tar.gz#egg=fr_core_news_sm",
            "de_core_news_sm @ https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.7.0/de_core_news_sm-3.7.0.tar.gz#egg=de_core_news_sm",
        ]
    },
)
