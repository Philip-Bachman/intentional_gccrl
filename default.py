"""Default logger."""

import logging
from typing import Any, Callable, Mapping, Optional

from acme.utils import paths
from acme.utils.loggers import NoOpLogger
from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import csv
from acme.utils.loggers import filters
from acme.utils.loggers import terminal
from acme.utils.loggers import tf_summary


def make_default_logger(
    label: str,
    save_data: bool = True,
    save_dir: str = 'logs',
    add_uid: bool = True,
    time_delta: float = 10.0,
    asynchronous: bool = False,
    print_fn: Optional[Callable[[str], None]] = None,
    serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = base.to_numpy,
    steps_key: str = 'steps',
    use_term: bool = False,
    use_tboard: bool = False,
) -> base.Logger:
  """Makes a default Acme logger.

  Args:
    label: Name to give to the logger.
    save_data: Whether to persist data.
    time_delta: Time (in seconds) between logging events.
    asynchronous: Whether the write function should block or not.
    print_fn: How to print to terminal (defaults to print).
    serialize_fn: An optional function to apply to the write inputs before
      passing them to the various loggers.
    steps_key: key for field that indicates steps (ie, x axis) for tboard plots
    use_term: bool = False,
    use_tboard: whether to push plots to tensorboard

  Returns:
    A logger object that responds to logger.write(some_dict).
  """
  if not print_fn:
    print_fn = logging.info

  # deal with adding uid and making useful logdir name to workaround
  # differences between different Acme loggers (namely, TFSummaryLogger)
  if add_uid:
    save_dir = paths.process_path(save_dir, 'logs', label, add_uid=True)
  
  loggers = []
  if use_term:
    loggers.append(terminal.TerminalLogger(label=label, print_fn=print_fn))
  if save_data:
    loggers.append(csv.CSVLogger(label=label, directory_or_file=save_dir))
  if use_tboard:
    loggers.append(
      tf_summary.TFSummaryLogger(label=label, logdir=save_dir, steps_key=steps_key)
    )
  if len(loggers) <= 0:
    print('WARNING -- creating some NoOpLoggers!!')
    loggers.append(NoOpLogger())

  # Dispatch to all writers and filter Nones and by time.
  logger = aggregators.Dispatcher(loggers, serialize_fn)
  logger = filters.NoneFilter(logger)
  if asynchronous:
    logger = async_logger.AsyncLogger(logger)
  logger = filters.TimeFilter(logger, time_delta)
  return logger
