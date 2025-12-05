"""EurlexResources"""

import json

import datasets

try:
    import lzma as xz
except ImportError:
    import pylzma as xz

datasets.logging.set_verbosity_info()
logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """
"""

_CITATION = """
"""

_URL = "https://huggingface.co/datasets/joelito/eurlex_resources"
_DATA_URL = f"{_URL}/resolve/main/data"

_LANGUAGES = [
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "ga",
    "hr",
    "hu",
    "it",
    "lt",
    "lv",
    "mt",
    "nl",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sv",
]

_RESOURCE_TYPES = ["caselaw", "decision", "directive", "intagr", "proposal", "recommendation", "regulation"]


class EurlexResourcesConfig(datasets.BuilderConfig):
    """BuilderConfig for EurlexResources."""

    def __init__(self, name: str, **kwargs):
        """BuilderConfig for EurlexResources.
        Args:
            name: combination of language and resource_type with _
            language: One of bg,cs,da,de,el,en,es,et,fi,fr,ga,hr,hu,it,lt,lv,mt,nl,pl,pt,ro,sk,sl,sv or all
            resource_type: One of caselaw, decision, directive, intagr, proposal, recommendation, regulation
          **kwargs: keyword arguments forwarded to super.
        """
        super(EurlexResourcesConfig, self).__init__(**kwargs)
        self.name = name
        self.language = name.split("_")[0]
        self.resource_type = name.split("_")[1]


class EurlexResources(datasets.GeneratorBasedBuilder):
    """EurlexResources: A Corpus Covering the Largest EURLEX Resources"""

    BUILDER_CONFIGS = [EurlexResourcesConfig(f"{language}_{resource_type}")
                       for resource_type in _RESOURCE_TYPES + ["all"]
                       for language in _LANGUAGES + ["all"]]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "celex": datasets.Value("string"),
                    "date": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_urls = []
        languages = _LANGUAGES if self.config.language == "all" else [self.config.language]
        resource_types = _RESOURCE_TYPES if self.config.resource_type == "all" else [self.config.resource_type]
        for language in languages:
            for resource_type in resource_types:
                data_urls.append(f"{_DATA_URL}/{language}/{resource_type}.jsonl.xz")

        downloaded_files = dl_manager.download(data_urls)
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": downloaded_files})]

    def _generate_examples(self, filepaths):
        """This function returns the examples in the raw (text) form by iterating on all the files."""
        id_ = 0
        for filepath in filepaths:
            logger.info("Generating examples from = %s", filepath)
            try:
                with xz.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                    for line in f:
                        if line:
                            example = json.loads(line)
                            if example is not None and isinstance(example, dict):
                                yield id_, {
                                    "celex": example.get("celex", ""),
                                    "date": example.get("date", ""),
                                    "title": example.get("title", ""),
                                    "text": example.get("text", ""),
                                }
                                id_ += 1
            except:
                print("Error reading file:", filepath)

