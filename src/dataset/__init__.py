from .SEN2VENUS_v2 import SEN2VENUSDataset
from .pelican_v2 import PelicanDataset
from .worldview import WorldView3Dataset

dict_datasets = {
    'pelican': PelicanDataset,
    'worldview': WorldView3Dataset,
    'sen2venus': SEN2VENUSDataset
}