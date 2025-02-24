from logging import Logger
from typing import Any, Callable

from fastapi import FastAPI
from utils.dataclasses import TaskParameters

from api.tasks import create_router


def create_app(
    task_params: dict[int, TaskParameters], func: Callable[..., Any], logger: Logger
) -> FastAPI:
    """
    Creates a FastAPI application and registers a router for task handling.

    Parameters
    ----------
    task_params : dict[int, TaskParameters]
        A dictionary mapping task IDs (int) to `TaskParameters` instances, defining
        the parameters for each task.
    func : Callable[..., Any]
        A callable function to be executed for task processing.
    logger : Logger
        A `Logger` instance for logging application events and task activity.

    Returns
    -------
    FastAPI
        A configured FastAPI application instance.

    Notes
    -----
    - The function uses `create_router` to generate a router with the specified
      `task_params`, `func`, and `logger`.
    - The created router is then included in the FastAPI application.
    """
    app = FastAPI()

    router = create_router(task_params, func, logger)
    app.include_router(router)

    return app
