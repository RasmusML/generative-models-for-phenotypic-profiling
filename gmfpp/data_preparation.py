import pandas as pd
import numpy as np

def read_metadata(path: str) -> pd.DataFrame:
    result = pd.read_csv(path)

    if not isinstance(result, pd.DataFrame):
        raise TypeError("error")

    return result

def save_metadata(metadata: pd.DataFrame, path: str):
    metadata.to_csv(path, index=False)

def filter_metadata_by_multi_cell_image_names(metadata: pd.DataFrame, multi_cell_image_names: list[str]) -> pd.DataFrame:
    """ Also corresponds to the subfolder name. """
    result = metadata[np.isin(metadata["Multi_Cell_Image_Name"], multi_cell_image_names)]

    if not isinstance(result, pd.DataFrame):
        raise TypeError("error")

    return result

def get_relative_image_paths(metadata: pd.DataFrame) -> list[str]:
    """ returns 'singh_cp_pipeline_singlecell_images'/subfolder/image_name """
    result = []

    multi_cell_image_name = metadata["Multi_Cell_Image_Name"]
    single_cell_image_id = metadata["Single_Cell_Image_Id"]

    if not isinstance(multi_cell_image_name, pd.Series):
        raise TypeError("error")

    if not isinstance(single_cell_image_id, pd.Series):
        raise TypeError("error")
    
    for multi_cell_name, image_id in zip(multi_cell_image_name, single_cell_image_id):
        path = "singh_cp_pipeline_singlecell_images/" + multi_cell_name + "/" + multi_cell_name + "_" + str(image_id) + ".npy"
        result.append(path)
        
    return result

def get_relative_image_folders(metadata: pd.DataFrame) -> list[str]:
    """ returns 'singh_cp_pipeline_singlecell_images'/subfolder/"""
    result = []

    multi_cell_image_name = metadata["Multi_Cell_Image_Name"]
    single_cell_image_id = metadata["Single_Cell_Image_Id"]

    if not isinstance(multi_cell_image_name, pd.Series):
        raise TypeError("error")

    if not isinstance(single_cell_image_id, pd.Series):
        raise TypeError("error")
    
    for multi_cell_name, image_id in zip(multi_cell_image_name, single_cell_image_id):
        folder = "singh_cp_pipeline_singlecell_images/" + multi_cell_name
        result.append(folder)
        
    return result


def load_images(paths: list[str]) -> list[np.ndarray]:
    result = []

    for path in paths:
        image = np.load(path)
        result.append(image)

    return result

def drop_redundant_metadata_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    " Drops unused and redundant columns "

    to_drop = ["Unnamed: 0", "Single_Cell_Image_Name", "Image_FileName_DAPI", "Image_PathName_DAPI", "Image_FileName_Tubulin", "Image_PathName_Tubulin", "Image_FileName_Actin", "Image_PathName_Actin"]
    result = metadata.drop(columns=to_drop)

    if not isinstance(result, pd.DataFrame):
        raise TypeError("error")

    return result
 
