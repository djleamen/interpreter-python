"""Token class for the Lox interpreter."""


class Token:
    """A simple Token class to represent tokens."""

    def __init__(self, token_type, lexeme, literal, line):
        self.type = token_type
        self.lexeme = lexeme
        self.literal = literal
        self.line = line

    def __repr__(self):
        return f"{self.type} {self.lexeme} {self.literal}"
