import json
import os


class Config(dict):
    """
    A dictionary-like configuration manager that allows loading, modifying,
    and saving configuration data from a JSON file.

    Parameters
    ----------
    path_to_cfg : str
        The path to the configuration JSON file.

    Attributes
    ----------
    _path : str
        The file path to the configuration JSON file.

    Methods
    -------
    upload() -> bool
        Saves the current configuration back to the file.

    Notes
    -----
    - The class initializes by loading the JSON data from the specified file
      and populating the dictionary with its contents.
    - Changes made to the configuration can be saved to the file using the
      `upload` method.
    """

    def __init__(self, path_to_cfg: str) -> None:
        """
        Initializes the Config object by loading data from the specified JSON file.

        Parameters
        ----------
        path_to_cfg : str
            The file path to the configuration JSON file.
        """
        self._path = path_to_cfg
        with open(self._path, "r", encoding="utf-8") as cfg_file:
            data = json.load(cfg_file)
        super().__init__(data)

    def upload(self) -> bool:
        """
        Saves the current configuration back to the file.

        Returns
        -------
        bool
            Returns `True` if the configuration is successfully saved, otherwise `False`.

        Notes
        -----
        - The method writes the current state of the dictionary to the JSON file
          specified by `_path`, formatted with an indentation of 4 spaces.
        - In case of an error (e.g., file permissions or disk issues), the method
          catches the exception and returns `False`.
        """
        try:
            with open(self._path, "w", encoding="utf-8") as cfg_file:
                json.dump(obj=self, fp=cfg_file, indent=4)
            return True
        except Exception:
            return False


general_cfg = Config(os.path.join(os.path.dirname(__file__), "general.json"))
