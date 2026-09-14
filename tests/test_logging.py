
import logging
import inspect

from logging.handlers import RotatingFileHandler


def remove_handlers(logger):
    """Remove and close all handlers from a logger."""
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


def test_log_message_and_line_number(tmp_path):
    """
    Verify that an INFO message is written to a file
    and includes the source filename and exact line number.
    """

    log_file = tmp_path / "test.log"

    logger = logging.getLogger("test_logging_message")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    remove_handlers(logger)

    handler = RotatingFileHandler(
        log_file,
        maxBytes=1024,
        backupCount=2,
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    try:
        expected_line = inspect.currentframe().f_lineno + 1
        logger.info("Test logging message")

        handler.flush()

        content = log_file.read_text(encoding="utf-8")

        assert "INFO" in content
        assert "Test logging message" in content
        assert f"test_logging.py:{expected_line}" in content

    finally:
        logger.removeHandler(handler)
        handler.close()


def test_exception_traceback(tmp_path):
    """
    Verify that logger.exception() records the error
    and its traceback.
    """

    log_file = tmp_path / "exception.log"

    logger = logging.getLogger("test_exception")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    remove_handlers(logger)

    handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    try:
        try:
            1 / 0
        except ZeroDivisionError:
            logger.exception("Division error occurred")

        handler.flush()

        content = log_file.read_text(encoding="utf-8")

        assert "ERROR" in content
        assert "Division error occurred" in content
        assert "Traceback (most recent call last)" in content
        assert "ZeroDivisionError" in content
        assert "division by zero" in content

    finally:
        logger.removeHandler(handler)
        handler.close()


def test_log_rotation(tmp_path):
    """
    Verify that RotatingFileHandler creates backup files
    and respects the backupCount limit.
    """

    log_file = tmp_path / "rotation.log"

    logger = logging.getLogger("test_rotation")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    remove_handlers(logger)

    handler = RotatingFileHandler(
        log_file,
        maxBytes=200,
        backupCount=2,
        encoding="utf-8",
    )

    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    try:
        for i in range(100):
            logger.info(
                "Rotation test message %s - %s",
                i,
                "x" * 50,
            )

        handler.flush()

        assert log_file.exists()

        backup_files = list(tmp_path.glob("rotation.log.*"))

        assert len(backup_files) >= 1
        assert len(backup_files) <= 2

    finally:
        logger.removeHandler(handler)
        handler.close()