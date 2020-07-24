from dataclasses import dataclass
from os import path

import nlp


@dataclass
class BratConfig(nlp.BuilderConfig):
    """BuilderConfig for BRAT."""

    ann_file_extension: str = 'ann'
    txt_file_extension: str = 'txt'


class Brat(nlp.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = BratConfig

    def _info(self):
        return nlp.DatasetInfo(
            features=nlp.Features(
                {
                    "context": nlp.Value("string"),
                    "spans": nlp.Sequence({
                        "id": nlp.Value("string"),
                        "type": nlp.Value("string"),
                        "locations": nlp.Sequence({
                            "start": nlp.Value("int"),
                            "end": nlp.Value("int"),
                        }),
                        "text": nlp.Value("string"),
                    }),
                    "relations": nlp.Sequence({
                        "id": nlp.Value("string"),
                        "type": nlp.Value("string"),
                        "arguments": nlp.Sequence({
                            "type": nlp.Value("string"),
                            "target": nlp.Value("string")
                        })
                    }),
                    "events": nlp.Sequence({
                        "id": nlp.Value("string"),
                        "type": nlp.Value("string"),
                        "trigger": nlp.Value("string"),
                        "arguments": nlp.Sequence({
                            "type": nlp.Value("string"),
                            "target": nlp.Value("string")
                        })
                    }),
                    # TODO: add attributions
                    # TODO: add normalizations
                })
        )

    def _split_generators(self, dl_manager):
        """ The `datafiles` kwarg in load_dataset() can be a str, List[str], Dict[str,str], or Dict[str,List[str]].

            If str or List[str], then the dataset returns only the 'train' split.
            If dict, then keys should be from the `nlp.Split` enum.
        """
        if isinstance(self.config.data_files, (str, list, tuple)):
            # Handle case with only one split
            files = self.config.data_files
            if isinstance(files, str):
                files = [files]
            return [nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"files": files})]
        else:
            # Handle case with several splits and a dict mapping
            splits = []
            for split_name in [nlp.Split.TRAIN, nlp.Split.VALIDATION, nlp.Split.TEST]:
                if split_name in self.config.data_files:
                    files = self.config.data_files[split_name]
                    if isinstance(files, str):
                        files = [files]
                    splits.append(nlp.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
            return splits

    @staticmethod
    def _get_location(location_string):
        parts = location_string.split(' ')
        assert len(parts) == 2, f'Wrong number of entries in location string. Expected 2, but found: {parts}'
        return {'start': int(parts[0]), 'end': int(parts[1])}

    @staticmethod
    def _get_span_annotation(annotation_text):
        res = {}
        _id, _remaining = annotation_text.split('\t', maxsplit=1)
        res['id'] = _id
        _annot, _text = _remaining.split('\t', maxsplit=1)
        res['text'] = _text
        _type, _locations = _annot.split(' ', maxsplit=1)
        res['type'] = _type
        res['locations'] = [Brat._get_location(loc) for loc in _locations.split(';')]
        
        return res

    @staticmethod
    def _get_event_annotation(annotation_text):
        raise NotImplementedError('implement _get_event_annotation()!')

    @staticmethod
    def _get_relation_annotation(annotation_text):
        raise NotImplementedError('implement _get_relation_annotation()!')

    @staticmethod
    def _get_attribute_annotation(annotation_text):
        raise NotImplementedError('implement _get_attribute_annotation()!')

    @staticmethod
    def _get_normalization_annotation(annotation_text):
        raise NotImplementedError('implement _get_normalization_annotation()!')

    @staticmethod
    def _read_annotation_file(filename):
        """
        reads a BRAT annotations file (see https://brat.nlplab.org/standoff.html)
        """

        res = {
            'spans': [],
            'events': [],
            'relations': [],
            'attributions': [],
            'normalizations': [],
        }

        with open(filename) as file:
            for line in file:
                # remove leading whitespace
                line = line.lstrip()
                if len(line) == 0 or line[0] == '#':
                    continue
                elif line[0] == 'T':
                    res['spans'].append(Brat._get_span_annotation(line))
                elif line[0] == 'E':
                    res['events'].append(Brat._get_event_annotation(line))
                elif line[0] == 'R':
                    res['relations'].append(Brat._get_relation_annotation(line))
                elif line[0] == 'A':
                    res['attributions'].append(Brat._get_attribute_annotation(line))
                elif line[0] == 'N':
                    res['normalizations'].append(Brat._get_normalization_annotation(line))
                else:
                    raise ValueError(f'unknown BRAT id type: {line[0]}. Annotation ids have to start with T (spans), '
                                     f'E (events), R (relations), A (attributions), or N (normalizations). See '
                                     f'https://brat.nlplab.org/standoff.html for the BRAT annotation file '
                                     f'specification.')
        return res

    def _generate_examples(self, files):
        """ Read files sequentially, then lines sequentially. """
        #idx = 0
        for filename in files:
            basename = path.basename(filename)

            ann_fn = f'{filename}.{self.config.ann_file_extension}'
            brat_annotations = Brat._read_annotation_file(ann_fn)

            txt_fn = f'{filename}.{self.config.txt_file_extension}'
            txt_content = open(txt_fn).read()
            brat_annotations['context'] = txt_content

            yield basename, brat_annotations
