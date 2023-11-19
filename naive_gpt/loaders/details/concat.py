from torch.utils import data
from torchdata import datapipes as dp


class ConcatDataset(data.IterableDataset):
    def __init__(self,
                 datasets: dict,
                 buffer_size: int = 1024,
                 infinite: bool = True):
        data.IterableDataset.__init__(self)
        #
        if infinite:
            datasets = {
                dp.iter.Cycler(source): weight
                for source, weight in datasets.items()
            }
        source_dp = dp.iter.SampleMultiplexer(
            pipes_to_weights_dict=datasets
        )
        source_dp = dp.iter.Shuffler(
            source_dp, buffer_size=buffer_size
        )
        self._source_dp = source_dp

    def __iter__(self):
        return iter(self._source_dp)
