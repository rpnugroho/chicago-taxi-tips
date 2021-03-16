import os
import urllib.request
import logging
import pandas as pd
import numpy as np

DATASET_URL = "https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/chicago_taxi_pipeline/data/simple/data.csv"
LOCAL_FILE_NAME = "dataset/chicago_taxi.csv"


def download_dataset(url=DATASET_URL, local_file_name=LOCAL_FILE_NAME):
    """Download dataset.

    Keyword Arguments:
        url {string} -- 
            Complete url path to download dataset. (default: {DATASET_URL})
        local_file_name {string} -- 
            Initial local file location. (default: {LOCAL_FILE_NAME})
    """
    urllib.request.urlretrieve(url, local_file_name)
    logging.info("Download completed.")


def check_execution_path():
    """Check if the function and therefore all subsequent functions
        are executed from the root of the project

    Returns:
        boolean -- returns False if execution path isn't the root,
            otherwise True
    """
    directory = "dataset/"
    if not os.path.exists(directory):
        logging.error(
            "Don't execute the script from a sub-directory."
            "Switch to the root of the project folder"
        )
        return False
    return True


def modify_dataset(local_file_name=LOCAL_FILE_NAME):
    """Create target label, if someone got tips > 0.2*fare then target label is 1.

    Keyword Arguments:
        local_file_name {string} -- Local file location (default: {LOCAL_FILE_NAME})
    """
    df = pd.read_csv(local_file_name)
    df["target"] = np.where(df["tips"] > (0.2*df["fare"]), 1, 0)
    df.drop("tips", axis=1, inplace=True)
    df.to_csv(local_file_name, index=None)
    logging.info("Target label created.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Started download script")

    if check_execution_path():
        download_dataset()
        modify_dataset()

    logging.info("Finished download script")
