
from __future__ import absolute_import, division, print_function

import json
import os


from dataclasses import dataclass
from os import path

import nlp

########################################################################################################################
### taken from https://github.com/ArneBinder/nlp/blob/brat/datasets/brat/brat.py                                     ###
########################################################################################################################


########################################################################################################################
from datasets.brat.brat import BratConfig, Brat

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

# TODO: Add description of the dataset here
_DESCRIPTION = """\
This dataset is an extension of the Dr. Inventor corpus (Fisas et al., 2015, 2016) with an annotation layer containing 
fine-grained argumentative components and relations. It is the first argument-annotated corpus of scientific 
publications (in English), which allows for joint analyses of argumentation and other rhetorical dimensions of 
scientific writing.
"""

_URL = "http://data.dws.informatik.uni-mannheim.de/sci-arg/compiled_corpus.zip"


# Using a specific configuration class is optional, you can also use the base class if you don't need
# to add specific attributes.
# here we give an example for three sub-set of the dataset with difference sizes.
class SciargConfig(BratConfig):
    """ BuilderConfig for SciArg"""


class Sciarg(Brat):

    VERSION = nlp.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.
    BUILDER_CONFIG_CLASS = SciargConfig
    #BUILDER_CONFIGS = [
    #    NewDatasetConfig(name="my_dataset_" + size, description="A small dataset", data_size=size) for size in ["small", "medium", "large"]
    #]

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
        data_dir = os.path.join(dl_dir, "sci-arg")
        print(f'data_dir: {data_dir}')
        return [
            nlp.SplitGenerator(
                name=nlp.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "directory": data_dir,
                    #'labelpath': os.path.join(data_dir, 'train_{}-labels.lst'.format(self.config.data_size)),
                    #"split": "train",
                },
            ),
        ]
