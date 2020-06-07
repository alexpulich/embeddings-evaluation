# import urllib.request
import requests
import os
import os.path as op
import logging

from config import DATASETS_DOWNLOAD_DIR

logger = logging.getLogger(__name__)


class DatasetDownloader:
    def __init__(self):
        if not op.exists(DATASETS_DOWNLOAD_DIR):
            os.mkdir(DATASETS_DOWNLOAD_DIR)

    def download_file(self, url: str, filename: str, tries: int = 3) -> str:
        path_to_download = op.join(DATASETS_DOWNLOAD_DIR, filename)

        if op.exists(path_to_download):
            logger.info(f'{path_to_download} exists, skipping.')
            return path_to_download

        for i in range(tries):
            try:
                # path, message = urllib.request.urlretrieve(url, path_to_download)
                response = requests.get(url)
                # TODO verify download, validate formats
                with open(path_to_download, 'wb') as file:
                    file.write(response.content)
                return path_to_download
            except Exception as e:
                logger.error(e)
                continue



dataset_downloader = DatasetDownloader()
