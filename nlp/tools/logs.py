"""Logging utilities."""

import sys
import logging
import colorlog


def config_logger(name, format='%(message)s', datefmt=None,
                  stream=sys.stdout, colored=False, level=logging.INFO,
                  filename=None, filemode='w', filelevel=None,
                  propagate=False):
    """Basic configuration for the logging system.

     Similar to logging.basicConfig but the logger `name` is configurable and
     both a file output and a stream output can be created. Returns a logger
     object.

    The default behaviour is to create a StreamHandler which writes to
    sys.stdout, set a formatter using the format string, and add the handler to
    the `name` logger.

    :parameters:

        :name:      Logger name
        :format:    handler format string (default=``%(message)s``)
        :datefmt:   handler date/time format specifier
        :stream:    initialize the StreamHandler using ``stream``
                    (None disables the stream, default=``sys.stdout``)
        :colored:   colored handler
        :level:     logger level (default=``INFO``).
        :filename:  create FileHandler using ``filename`` (default=``None``)
        :filemode:  open ``filename`` with specified filemode (``w`` or ``a``)
        :filelevel: logger level for file logger (default=``level``)
        :propagate: propagate message to parent (default=``False``)
    """
    # Get a logger for the specified name
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    # Remove existing handlers, otherwise multiple handlers can accrue
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    # Add handlers. Add NullHandler if no file or stream output so that
    # modules don't emit a warning about no handler.
    if not (filename or stream or colored):
        logger.addHandler(logging.NullHandler())

    if filename:
        hdlr = logging.FileHandler(filename, filemode)
        if filelevel is None:
            filelevel = level
            hdlr.setLevel(filelevel)

    if stream:
        hdlr = logging.StreamHandler(stream)
        hdlr.setLevel(level)

    if colored:
        hdlr = colorlog.StreamHandler()
        hdlr.setFormatter(colorlog.ColoredFormatter('%(log_color)s' + format,
                                                    log_colors={
                                                        'DEBUG':    'cyan',
                                                        'INFO':     'green',
                                                        'WARNING':  'yellow',
                                                        'ERROR':    'red',
                                                        'CRITICAL': 'bg_white',
                                                    }))
    else:
        hdlr.setFormatter(logging.Formatter(format))
    logger.addHandler(hdlr)

    return logger
