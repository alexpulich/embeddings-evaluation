from enum import Enum

import numpy as np
import pandas as pd

from sklearn.utils import Bunch

from .downloader import dataset_downloader


class DatasetEnum(Enum):
    WORDSIM = 'WORDSIM'
    SEMEVAL = 'SEMEVAL'
    TWS = 'TWS'
    SIMLEX = 'SIMLEX'


class DatasetLoader:
    _datasets_urls = {
        'WORDSIM': 'https://www.dropbox.com/s/h8c3ll1764d7akf/thai-wordsim353-v2.csv?dl=1',
        'SEMEVAL': 'https://www.dropbox.com/s/scfopjmis59s7c3/thaiSemEval-500-v2.csv?dl=1',
        'TWS': 'https://www.dropbox.com/s/qtiys0c17dmnywj/tws65.csv?dl=1',
        'SIMLEX': 'https://www.dropbox.com/s/nlct64af7qmhc49/thaiSimLex-999-v2.csv?dl=1'
    }

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset_path = self._download_dataset()

    def _download_dataset(self) -> str:
        dataset_url = self._datasets_urls.get(self.dataset_name)
        assert dataset_url is not None, f'N' \
                                        f'o download link for {self.dataset_name}'
        dataset_path = dataset_downloader.download_file(dataset_url, f'{self.dataset_name}.csv')
        if not dataset_path:
            raise RuntimeError('Failed to download the dataset')

        return dataset_path

    def get_data(self):
        data = pd.read_csv(self.dataset_path, header=None, sep=',').values

        return Bunch(X=data[:, 0:2].astype("object"),
                     y=2 * data[:, 2].astype(np.float))


tasks = (
    DatasetEnum.WORDSIM.value,
    DatasetEnum.SEMEVAL.value,
    DatasetEnum.SIMLEX.value,
    DatasetEnum.TWS.value,
)
