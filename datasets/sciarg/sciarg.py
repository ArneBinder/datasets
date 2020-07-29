
from __future__ import absolute_import, division, print_function

import os

import nlp

from datasets.brat.brat import AbstractBrat

_CITATION = """
@inproceedings{lauscher2018b,
  title = {An argument-annotated corpus of scientific publications},
  booktitle = {Proceedings of the 5th Workshop on Mining Argumentation},
  publisher = {Association for Computational Linguistics},
  author = {Lauscher, Anne and Glava\v{s}, Goran and Ponzetto, Simone Paolo},
  address = {Brussels, Belgium},
  year = {2018},
  pages = {40–46}
}
"""

_DESCRIPTION = """\
This dataset is an extension of the Dr. Inventor corpus (Fisas et al., 2015, 2016) with an annotation layer containing 
fine-grained argumentative components and relations. It is the first argument-annotated corpus of scientific 
publications (in English), which allows for joint analyses of argumentation and other rhetorical dimensions of 
scientific writing.
"""

_URL = "http://data.dws.informatik.uni-mannheim.de/sci-arg/compiled_corpus.zip"


class Sciarg(AbstractBrat):

    VERSION = nlp.Version("1.0.0")

    def _info(self):

        brat_features = super()._info().features
        return nlp.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # nlp.features.FeatureConnectors
            features=brat_features,
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            #supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/anlausch/ArguminSci",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: Downloads the data and defines the splits
        # dl_manager is a nlp.download.DownloadManager that can be used to
        # download and extract URLs
        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, "compiled_corpus")
        print(f'data_dir: {data_dir}')
        return [
            nlp.SplitGenerator(
                name=nlp.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "directory": data_dir,
                },
            ),
        ]
