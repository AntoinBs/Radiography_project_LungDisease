import pandas as pd
from pathlib import Path

def add_url(df : pd.DataFrame, file_type : str = 'img') -> None:
    r"""
    This function update the DataFrame df with the images or masks url, relatively to the project root.
    args:
        df : pd.DataFrame - Dataframe where the path will be added
        file_type : str - ['img', 'mask'] default : 'img'. Indicates if you want to add images or masks url in the DataFrame
    returns:
        None
    """
    if file_type not in ['img', 'mask']: # Checking that file_type is in accepted values
        raise ValueError(f"Le paramètre file_type={file_type} n'est pas correct. Valeurs acceptées : ['img', 'mask']")
    elif file_type == 'img':
        col = 'IMG_URL' # Updated column of df will be 'IMG_URL'
        folder = "images" # URL will contains the folder 'image'
    elif file_type == 'mask':
        col = 'MASK_URL'
        folder = "masks"
    
    for file, format, label in zip(df['FILE NAME'], df['FORMAT'], df['LABEL']):
        format = format.lower() # set format in lowercase
        file_name = file + '.' + format # concatenate file and format to make global file_name
        path = Path(r"..","data", "raw", label, folder, file_name) # build Path from the variables in the current line of dataset
        df.loc[(df['FILE NAME'] == file), col] = path # update the path in DataFrame
        print('Le fichier ',file, 'a été ajouté au tableau.')
    return df