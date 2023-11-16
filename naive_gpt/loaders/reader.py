import os
import torch
import random
from torch.utils import data
from torchdata import datapipes as dp
from naive_gpt import loaders


class LineReader(data.IterableDataset):
    def __init__(self,
                 root: str,
                 files: dict,
                 reader: str = 'line',
                 shuffle: bool = True,
                 skip_lines: int = 0,
                 min_length: int = 64,
                 buffer_size: int = 16384,
                 return_path: bool = False,
                 append_path: bool = False,
                 text_transform: callable = None,
                 path_transform: callable = None):
        data.IterableDataset.__init__(self)
        #
        self.skip_lines = skip_lines
        self.min_length = min_length
        self.return_path = return_path
        self.append_path = append_path
        self.text_transform = text_transform
        self.path_transform = path_transform
        #
        weighted_dp = {}
        for path, weight in files.items():
            path = '{}/{}'.format(
                root.rstrip('/'), path.lstrip('/')
            )
            source = dp.iter.FileOpener([path])
            # reader
            if reader == 'csv':
                source = dp.iter.CSVParser(
                    source, return_path=True,
                    skip_lines=skip_lines
                )
            elif reader == 'line':
                source = dp.iter.LineReader(
                    source, return_path=True,
                    skip_lines=skip_lines
                )
            else:
                raise RuntimeError
            #
            source = dp.iter.Cycler(source)
            weighted_dp[source] = weight
        #
        source_dp = dp.iter.SampleMultiplexer(
            pipes_to_weights_dict=weighted_dp
        )
        source_dp = dp.iter.Mapper(
            source_dp, fn=self._clean_fn
        )
        source_dp = dp.iter.Filter(
            source_dp, filter_fn=self._filter_fn
        )
        if shuffle:
            source_dp = dp.iter.Shuffler(
                source_dp, buffer_size=buffer_size
            )
        self._source_dp = source_dp

    def _clean_fn(self, item: list):
        assert len(item) == 2
        filename, content = item
        #
        if isinstance(content, str):
            content = loaders.Sanitize()(content)
        elif isinstance(content, (list, tuple)):
            content = [
                loaders.Sanitize()(item) for item in content
            ]
        else:
            raise RuntimeError
        #
        return filename, content

    def _filter_fn(self, item: list):
        assert len(item) == 2
        _, content = item
        #
        if isinstance(content, str):
            n_size = len(content)
        elif isinstance(content, (list, tuple)):
            n_size = sum([
                len(item) for item in content
            ])
        else:
            raise RuntimeError
        #
        if n_size < self.min_length:
            return False
        return True

    def __iter__(self):
        seed = int.from_bytes(
            os.urandom(4),
            byteorder='little'
        )
        random.seed(seed)
        torch.random.manual_seed(seed)
        #
        for item in self._source_dp:
            filename, content = item
            # append
            if self.append_path:
                content = (content, filename)
            # transform
            if self.text_transform is not None:
                content = self.text_transform(content)
            if self.path_transform is not None:
                filename = self.path_transform(filename)
            if self.return_path:
                yield content, filename
            else:
                yield content


class TextFolder(LineReader):
    def __init__(self,
                 root: str,
                 reader: str = 'line',
                 shuffle: bool = True,
                 skip_lines: int = 0,
                 min_length: int = 64,
                 buffer_size: int = 16384,
                 return_path: bool = True,
                 append_path: bool = False,
                 text_transform: callable = None,
                 path_transform: callable = None):
        #
        files = {
            filename.removeprefix(root): os.stat(filename).st_size
            for filename in dp.iter.FileLister(root=root)
        }
        LineReader.__init__(
            self, root=root, files=files, reader=reader,
            shuffle=shuffle, skip_lines=skip_lines, min_length=min_length,
            buffer_size=buffer_size, return_path=return_path,
            append_path=append_path, text_transform=text_transform,
            path_transform=path_transform
        )
