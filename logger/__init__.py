import json
import logging
import logging.config
import os


def create_logger(name: str) -> logging.Logger:
    """
    Creates and configures a logger instance based on a predefined JSON configuration file.

    Parameters
    ----------
    name : str
        The name of the logger, which will also be used to name the log file.

    Returns
    -------
    logging.Logger
        A configured logger instance.

    Notes
    -----
    - The function loads the logging configuration from a `config.json` file
      located in the same directory as the script.
    - Updates the file handler's `filename` property to write logs to a file
      named `{name}.log` inside a `logs` directory.
    - Ensures the `logs` directory exists; creates it if it does not.
    - Applies the logging configuration using `logging.config.dictConfig`.
    """
    logger_config = os.path.join(
        os.path.dirname(__file__),
        "config.json"
    )

    with open(logger_config, encoding="utf-8") as conf_file:
        config = json.load(conf_file)

    config["handlers"]["file"]["filename"] = f"logs/{name}.log"

    log_dir = os.path.join(os.path.abspath(os.path.curdir), "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logging.config.dictConfig(config)

    return logging.getLogger()
