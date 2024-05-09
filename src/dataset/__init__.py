from .SEN2VENUS import SEN2VENUSDataset
from .pelican import PelicanDataset
from .worldview import WorldView3Dataset

dict_datasets = {
    'pelican': PelicanDataset,
    'worldview': WorldView3Dataset,
    'sen2venus': SEN2VENUSDataset
}