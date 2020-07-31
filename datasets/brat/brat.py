
import nlp

from nlp_formats.brat import AbstractBrat


class Brat(AbstractBrat):

    def _split_generators(self, dl_manager):
        """ The `datafiles` kwarg in load_dataset() can be a str, List[str], Dict[str,str], or Dict[str,List[str]].

            If str or List[str], then the dataset returns only the 'train' split.
            If dict, then keys should be from the `nlp.Split` enum.
        """
        if self.config.data_dir is not None:
            return [nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"directory": self.config.data_dir})]
        else:
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
