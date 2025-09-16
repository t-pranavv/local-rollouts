Logging
=======

PhyAGI utilizes the ``logging`` package to provide a flexible and efficient logging mechanism for its components. Logging is crucial for tracking runtime behavior, monitoring performance, and troubleshooting issues.

Logger
------

A :class:`.Logger` is the primary object for issuing logging events in applications and libraries. It serves as the entry point for the logging mechanism and provides an interface for capturing and processing log messages. A separate logger instance is created for each module to ensure isolation and granularity. Loggers support various log levels, such as ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, and ``CRITICAL``, which represent increasing severity levels for log messages.

Filters
-------

Filters allow selective processing of log records based on specific criteria. They can be attached to loggers or handlers to control which records are passed through the logging pipeline.

In PhyAGI, a custom filter, :class:`._RankFilter`, manages log records from different processes in a parallel computing environment. It checks the local and MPI ranks of each log record, allowing logging only from the main process (i.e., the process with rank 0). This prevents duplicate log messages from multiple processes and ensures clean, analyzable log output.

Handlers
--------

Handlers dispatch log records to various output destinations, such as files, consoles, network sockets, or email servers. Each handler instance is associated with a specific output destination and processes log records from loggers or filters.

PhyAGI includes several custom handlers for various logging requirements:

* :class:`.StreamHandler`: Prints log messages to standard output in real-time.

* :class:`.TimedRotatingFileHandler`: Writes log records to a ``.log`` file, with automatic rotation based on a time interval to prevent excessive file size.

* :class:`.MlflowHandler`: Sends log records to MLflow, a popular platform for managing machine learning experiments. This handler is enabled by default only within trainer classes, requiring no additional configuration as AzureML/Amulet sets the necessary run and experiment identifiers.

* :class:`.WandbHandler`: Sends log records to Weights & Biases (WandB), a platform for tracking and visualizing machine learning experiments. The :class:`.WandbHandler` is enabled by default within trainer classes. To configure, two environment variables are required:

  * ``WANDB_API_KEY``: The personal API key, which can be provided via an environment variable or the ``key`` argument in the class constructor.
  * ``WANDB_HOST``: For Microsoft Research users, set this to ``https://microsoft-research.wandb.io`` via an environment variable or the ``host`` argument in the class constructor.

These handlers are designed to integrate seamlessly with machine learning platforms, offering a simple, user-friendly experience. Users can enable additional features as needed.

Using the handlers
------------------

By default, the :class:`.MlflowHandler` and :class:`.WandbHandler` are enabled when using the :class:`.DsTrainer` class. However, they can be disabled by setting the ``mlflow`` and ``wandb`` training arguments to ``False``.

Training arguments, such as :class:`.DsTrainingArguments`, can define the ``wandb_api_key`` and ``wandb_host`` arguments for WandB configuration. Alternatively, the environment variables ``WANDB_API_KEY`` and ``WANDB_HOST`` can be set. Mlflow is automatically configured by AzureML/Amulet.

Additionally, these handlers can be used independently. Below is an example of using the :class:`.MlflowHandler` and :class:`.WandbHandler` to log a dictionary of metrics:

Example
-------

.. code-block:: python

   from phyagi.utils.logging_handlers import MlflowHandler, WandbHandler
   from phyagi.utils.logging_utils import get_logger

   mlflow_handler = MlflowHandler()
   wandb_handler = WandbHandler(key=None, host=None, config={})

   logger = logging.getLogger(__name__)
   logger.addHandler(mlflow_handler)
   logger.addHandler(wandb_handler)

   logger.info({"loss": 0.0, "time": 0.0})

- Ensure that the appropriate environment variables are set for WandB configuration.
- The handlers are automatically configured when using :class:`.DsTrainer`.