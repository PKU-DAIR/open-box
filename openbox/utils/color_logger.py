"""
Author: Huaijun Jiang
Date: 2022-02-21 (Last Update: 2022-12-07)

Color Logger
+ Display color logging information on stream (console) using ColorFormatter
+ Support cross-platform (Windows, Linux, macOS) and IO redirection using colorama

Usage:
    import color_logger as logger
    logger.init(name='logger_name', level='DEBUG', stream=True, logdir='logs/')
    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.critical('critical')
    try:
        raise ValueError('value error')
    except ValueError:
        logger.exception('exception')
"""

import os
import logging
from datetime import datetime
import colorama

# format string. see `logging.Formatter` for more details
DEFAULT_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s"

# color for each logging level
log_colors = {
    logging.DEBUG: colorama.Style.BRIGHT + colorama.Fore.BLUE,
    logging.INFO: colorama.Style.BRIGHT + colorama.Fore.GREEN,
    logging.WARNING: colorama.Style.BRIGHT + colorama.Fore.YELLOW,
    logging.ERROR: colorama.Style.BRIGHT + colorama.Fore.RED,
    logging.CRITICAL: colorama.Style.BRIGHT + colorama.Back.RED,
}


class ColorFormatter(logging.Formatter):
    """
    A logging Formatter that adds colors to specified attributes of the log record.
    """
    available_attrs = ['name', 'levelno', 'levelname', 'pathname', 'filename', 'module', 'lineno', 'funcName',
                       'created', 'asctime', 'msecs', 'relativeCreated', 'thread', 'threadName', 'process', 'message']
    default_color_attrs = ['levelname']
    default_bold_attrs = ['asctime', 'name', 'filename']

    def __init__(self, fmt, *, color_attrs=None, bold_attrs=None, **kwargs):
        super().__init__(fmt, **kwargs)
        self._color_attrs = color_attrs if color_attrs is not None else self.default_color_attrs
        self._bold_attrs = bold_attrs if bold_attrs is not None else self.default_bold_attrs
        self._check_attrs()

    def _check_attrs(self):
        for attrs in (self._color_attrs, self._bold_attrs):
            for attr in attrs:
                assert attr in self.available_attrs, 'Attr %s not available in ColorFormatter!' % (attr, )

    @staticmethod
    def update_record_attr(record, attr, func, *args):
        """Use `func` to update `record.attr`"""
        try:
            value = getattr(record, attr)
        except AttributeError:
            return
        new_value = func(value, *args)
        setattr(record, attr, new_value)

    @staticmethod
    def colorize(s, color):
        return color + s + colorama.Style.RESET_ALL

    def formatMessage(self, record: logging.LogRecord) -> str:
        # color attr according to log level
        if record.levelno in log_colors.keys():
            color = log_colors[record.levelno]
            for attr in self._color_attrs:
                self.update_record_attr(record, attr, self.colorize, color)

        # bold attr
        bold = colorama.Style.BRIGHT
        for attr in self._bold_attrs:
            self.update_record_attr(record, attr, self.colorize, bold)

        return super().formatMessage(record)


class ColorLogger(logging.Logger):
    """
    Display color logging information on stream (console) using ColorFormatter
    """

    def __init__(self, name, level='INFO', stream=True, logfile=None, fmt=DEFAULT_FORMAT, color=True):
        """
        Parameters
        ----------
        name: str
            Logger name
        level: str, default='INFO'
            Logging level. Should be in ['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        stream: bool, default=True
            Whether to display logging information on stdout/stderr.
        logfile: str, default=None
            Log file path. If None, no log file will be created.
        fmt: str, default=DEFAULT_FORMAT
            Format string of logging information. see `logging.Formatter` for more details
        color: bool, default=True
            Whether to display color information on stream (console)
        """
        super().__init__(name, level)
        if logfile:
            logdir = os.path.dirname(logfile)
            if logdir != '':
                os.makedirs(logdir, exist_ok=True)
            plain_formatter = logging.Formatter(fmt)
            file_handler = logging.FileHandler(filename=logfile, mode='a', encoding='utf8')
            file_handler.setFormatter(plain_formatter)
            self.addHandler(file_handler)
        if stream:
            formatter = ColorFormatter(fmt) if color else logging.Formatter(fmt)
            console_handler = logging.StreamHandler()
            # support both ANSI terminals (Linux, macOS) and Windows console, and handle IO redirection
            console_handler.setStream(colorama.AnsiToWin32(console_handler.stream).stream)
            console_handler.setFormatter(formatter)
            self.addHandler(console_handler)


# If you want to make all loggers created by `logging.getLogger(name)`
# use ColorLogger, uncomment the following line.
# This will affect other running projects using `logging` module.
# However, since only `name` is passed to `ColorLogger`, the logger
# created by `logging.getLogger(name)` will not save log file.
#
# logging.setLoggerClass(ColorLogger)


logger = None  # type: ColorLogger

# Logging Functions (Assign logger functions later.
#     Do not define wrapper functions to prevent logging.Logger.findCaller from tracing this file.)
debug = None
info = None
warning = None
warn = warning
error = None
exception = None
critical = None
setLevel = None


def init(name='OpenBox', level='INFO', stream=True, logdir=None,
         fmt=DEFAULT_FORMAT, color=True, force_init=True):
    """
    Init logger

    Parameters
    ----------
    name: str
        Logger name
    level: str, default='INFO'
        Logging level. Should be in ['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    stream: bool, default=True
        Whether to display logging information on stdout/stderr.
    logdir: str, default=None
        Log file directory. If None, no log file will be created.
    fmt: str, default=DEFAULT_FORMAT
        Format string of logging information. see `logging.Formatter` for more details
    color: bool, default=True
        Whether to display color information on stream (console)
    force_init: bool, default=True
        If False, the logger will not be initialized if it has already been initialized.
    """
    global logger

    # only init once if force_init is False
    if logger is not None and not force_init:
        return

    if logdir is None:
        logfile = None
    else:
        os.makedirs(logdir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        logfile = os.path.join(logdir, '%s_%s.log' % (name, timestamp))
        logfile = os.path.abspath(logfile)

    logger = ColorLogger(name=name, level=level, stream=stream, logfile=logfile, fmt=fmt, color=color)

    global debug, info, warning, warn, error, exception, critical, setLevel
    debug = logger.debug
    info = logger.info
    warning = logger.warning
    warn = warning
    error = logger.error
    exception = logger.exception
    critical = logger.critical
    setLevel = logger.setLevel

    debug('Logger init.')
    if logfile is not None:
        info('Logfile: %s' % (logfile, ))


# init a default logger that will not save log file
init('OpenBox', level='INFO', stream=True, logdir=None, fmt=DEFAULT_FORMAT, color=True, force_init=False)
