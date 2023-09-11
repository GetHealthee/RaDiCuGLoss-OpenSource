import json
import logging
import sys
import functools


class JSONLogger:
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            json_log_object = {
                "message": record.getMessage(),
                "level": record.levelname,
                "timestamp": self.formatTime(record, self.datefmt)
            }
            return json.dumps(json_log_object)

    def __init__(self, name, level='DEBUG'):
        self.logger = logging.getLogger(name)

        logging_level = getattr(logging, level.upper(), None)
        if not isinstance(logging_level, int):
            raise ValueError('Invalid log level: %s' % level)
        self.logger.setLevel(logging_level)

        handler = logging.StreamHandler(sys.stdout)

        formatter = self.JSONFormatter()
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def __getattr__(self, attr):
        return getattr(self.logger, attr)


def log_decorator(logger):
    """
    a decorator that accepts a logger and
    :param logger:
    :return:
    """
    class PrintCapture:
        def __init__(self, logger):
            self.logger = logger

        def write(self, s):
            s_strip = s.strip()
            if s_strip:  # Prevent logging empty lines
                self.logger.info(s_strip)

        def flush(self):
            pass

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(
                f'Function {func.__name__} called with positional arguments {args} and keyword arguments {kwargs}')
            original_stdout = sys.stdout
            try:
                sys.stdout = PrintCapture(logger)
                result = func(*args, **kwargs)
            finally:
                sys.stdout = original_stdout
            return result

        return wrapper

    return decorator

