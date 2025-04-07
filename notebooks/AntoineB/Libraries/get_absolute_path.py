from pathlib import Path

def get_repo_path(path : 'str', repo_name : 'str' = 'NOV24-BDS---Covid19-groupe-1') -> 'str':
    r"""
    This function return the repository path depending on its name 'repo_name'. This function is working only if the current program is running in child folders
    of repository.
    args:
        path: str - path including the repository name repo_name (default : 'NOV24-BDS---Covid19-groupe-1')
        repo_name: str - repository's name (default : 'NOV24-BDS---Covid19-groupe-1')
    returns:
        repo_path: str - repository's absolute path
    """
    parts = path.parts # split path in list of each folders names
    for i, part in enumerate(parts):
        if part == repo_name:
            return(Path(*parts[:i+1])) # return the new path through the repo_name
    raise ValueError(f'There is no repo directory {repo_name} (set by repo_name) in the path') # raise ValueError if no repo_name find in path

def get_absolute_path(path : str, repo_name : str = 'NOV24-BDS---Covid19-groupe-1') -> str:
    r"""
    This function return the absolute path of path which must be relative path from repo folder (ex: data\raw\COVID\images\COVID-1.png)
    args:
        path : str - relative path from repo folder(ex: 'data\raw\COVID\images\COVID-1.png')
        repo_name : str - repository's name (default : 'NOV24-BDS---Covid19-groupe-1')
    returns:
        absolute_path : str - absolute path of path
    """
    # get the folder in which the current program is 
    try :
        parent_folder_path = Path(__file__).resolve().parent # __file__ is accessible in .py files but not in .ipynb files
    except NameError:
        import os
        parent_folder_path = Path(os.getcwd()).resolve() # for .ipynb files
    repo_path = get_repo_path(path = parent_folder_path, repo_name = repo_name) # get the github repo root path on the local computer
    return repo_path / path # return the repo_path concatenate with path which is relative path from the repo