"""Custom exceptions for the Lox interpreter."""


class ParseException(Exception):
    """Exception raised during parsing."""


class LoxRuntimeError(Exception):
    """Exception raised during runtime evaluation."""

    def __init__(self, token, message):
        self.token = token
        self.message = message
        super().__init__(message)


class Return(Exception):
    """Exception used for return control flow."""

    def __init__(self, value):
        self.value = value
        super().__init__()
