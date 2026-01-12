"""
Code Interpreter - A simple interpreter in Python
From CodeCrafters.io build-your-own-interpreter (Python)
"""

import sys


class Token:
    """A simple Token class to represent tokens."""

    def __init__(self, token_type, lexeme, literal, line):
        self.type = token_type
        self.lexeme = lexeme
        self.literal = literal
        self.line = line

    def __repr__(self):
        return f"{self.type} {self.lexeme} {self.literal}"


# Expression classes
class Expr:
    """Base class for expressions."""
    pass


class Literal(Expr):
    """Literal expression."""

    def __init__(self, value):
        self.value = value


class Binary(Expr):
    """Binary expression."""

    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right


class Unary(Expr):
    """Unary expression."""

    def __init__(self, operator, right):
        self.operator = operator
        self.right = right


class Grouping(Expr):
    """Grouping expression."""

    def __init__(self, expression):
        self.expression = expression


class ParseException(Exception):
    """Exception raised during parsing."""
    pass


class AstPrinter:
    """Class to print the AST in a parenthesized format."""

    def print(self, expr):
        """Print the expression in parenthesized format."""
        if isinstance(expr, Literal):
            if expr.value is None:
                return "nil"
            elif isinstance(expr.value, bool):
                return "true" if expr.value else "false"
            elif isinstance(expr.value, float):
                return str(expr.value)
            elif isinstance(expr.value, str):
                return expr.value
        elif isinstance(expr, Binary):
            return self.parenthesize(expr.operator.lexeme, expr.left, expr.right)
        elif isinstance(expr, Unary):
            return self.parenthesize(expr.operator.lexeme, expr.right)
        elif isinstance(expr, Grouping):
            return self.parenthesize("group", expr.expression)
        return ""

    def parenthesize(self, name, *exprs):
        """Helper method to parenthesize expressions."""
        result = f"({name}"
        for expr in exprs:
            result += " " + self.print(expr)
        result += ")"
        return result


class Parser:
    """A simple recursive descent parser."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0

    def parse(self):
        """Parse the tokens and return the expression."""
        try:
            return self.expression()
        except (IndexError, ValueError):
            return None

    def expression(self):
        """Parse an expression."""
        return self.equality()

    def equality(self):
        """Parse equality expressions."""
        expr = self.comparison()
        while self.match("BANG_EQUAL", "EQUAL_EQUAL"):
            operator = self.previous()
            right = self.comparison()
            expr = Binary(expr, operator, right)
        return expr

    def comparison(self):
        """Parse comparison expressions."""
        expr = self.term()
        while self.match("GREATER", "GREATER_EQUAL", "LESS", "LESS_EQUAL"):
            operator = self.previous()
            right = self.term()
            expr = Binary(expr, operator, right)
        return expr

    def term(self):
        """Parse term expressions."""
        expr = self.factor()
        while self.match("MINUS", "PLUS"):
            operator = self.previous()
            right = self.factor()
            expr = Binary(expr, operator, right)
        return expr

    def factor(self):
        """Parse factor expressions."""
        expr = self.unary()
        while self.match("SLASH", "STAR"):
            operator = self.previous()
            right = self.unary()
            expr = Binary(expr, operator, right)
        return expr

    def unary(self):
        """Parse unary expressions."""
        if self.match("BANG", "MINUS"):
            operator = self.previous()
            right = self.unary()
            return Unary(operator, right)
        return self.primary()

    def primary(self):
        """Parse primary expressions."""
        if self.match("FALSE"):
            return Literal(False)
        if self.match("TRUE"):
            return Literal(True)
        if self.match("NIL"):
            return Literal(None)
        if self.match("NUMBER"):
            return Literal(self.previous().literal)
        if self.match("STRING"):
            return Literal(self.previous().literal)
        if self.match("LEFT_PAREN"):
            expr = self.expression()
            self.consume("RIGHT_PAREN", "Expect ')' after expression.")
            return Grouping(expr)
        raise ParseException("Expected expression.")

    def match(self, *types):
        """Check if the current token matches any of the given types."""
        for token_type in types:
            if self.check(token_type):
                self.advance()
                return True
        return False

    def check(self, token_type):
        """Check if the current token is of the given type."""
        if self.is_at_end():
            return False
        return self.peek().type == token_type

    def advance(self):
        """Advance to the next token and return the previous one."""
        if not self.is_at_end():
            self.current += 1
        return self.previous()

    def is_at_end(self):
        """Check if we have reached the end of the token list."""
        return self.peek().type == "EOF"

    def peek(self):
        """Return the current token without consuming it."""
        return self.tokens[self.current]

    def previous(self):
        """Return the most recently consumed token."""
        return self.tokens[self.current - 1]

    def consume(self, token_type, message):
        """Consume a token of the given type or raise an error."""
        if self.check(token_type):
            return self.advance()
        raise ParseException(message)


def tokenize(file_contents):
    """Tokenize the file contents and return list of tokens."""
    tokens = []
    has_error = False
    i = 0
    line = 1

    keywords = {
        'and': 'AND',
        'class': 'CLASS',
        'else': 'ELSE',
        'false': 'FALSE',
        'for': 'FOR',
        'fun': 'FUN',
        'if': 'IF',
        'nil': 'NIL',
        'or': 'OR',
        'print': 'PRINT',
        'return': 'RETURN',
        'super': 'SUPER',
        'this': 'THIS',
        'true': 'TRUE',
        'var': 'VAR',
        'while': 'WHILE'
    }

    # Tokenize the file contents
    while i < len(file_contents):
        char = file_contents[i]

        if char == '(':
            tokens.append(Token("LEFT_PAREN", "(", None, line))
            i += 1
        elif char == ')':
            tokens.append(Token("RIGHT_PAREN", ")", None, line))
            i += 1
        elif char == '{':
            tokens.append(Token("LEFT_BRACE", "{", None, line))
            i += 1
        elif char == '}':
            tokens.append(Token("RIGHT_BRACE", "}", None, line))
            i += 1
        elif char == ',':
            tokens.append(Token("COMMA", ",", None, line))
            i += 1
        elif char == '.':
            tokens.append(Token("DOT", ".", None, line))
            i += 1
        elif char == '-':
            tokens.append(Token("MINUS", "-", None, line))
            i += 1
        elif char == '+':
            tokens.append(Token("PLUS", "+", None, line))
            i += 1
        elif char == ';':
            tokens.append(Token("SEMICOLON", ";", None, line))
            i += 1
        elif char == '*':
            tokens.append(Token("STAR", "*", None, line))
            i += 1
        elif char == '/':
            if i + 1 < len(file_contents) and file_contents[i + 1] == '/':
                i += 2
                while i < len(file_contents) and file_contents[i] != '\n':
                    i += 1
            else:
                tokens.append(Token("SLASH", "/", None, line))
                i += 1
        elif char == '=':
            if i + 1 < len(file_contents) and file_contents[i + 1] == '=':
                tokens.append(Token("EQUAL_EQUAL", "==", None, line))
                i += 2
            else:
                tokens.append(Token("EQUAL", "=", None, line))
                i += 1
        elif char == '!':
            if i + 1 < len(file_contents) and file_contents[i + 1] == '=':
                tokens.append(Token("BANG_EQUAL", "!=", None, line))
                i += 2
            else:
                tokens.append(Token("BANG", "!", None, line))
                i += 1
        elif char == '<':
            if i + 1 < len(file_contents) and file_contents[i + 1] == '=':
                tokens.append(Token("LESS_EQUAL", "<=", None, line))
                i += 2
            else:
                tokens.append(Token("LESS", "<", None, line))
                i += 1
        elif char == '>':
            if i + 1 < len(file_contents) and file_contents[i + 1] == '=':
                tokens.append(Token("GREATER_EQUAL", ">=", None, line))
                i += 2
            else:
                tokens.append(Token("GREATER", ">", None, line))
                i += 1
        elif char == '"':
            # String literal
            start = i
            i += 1
            string_line = line
            while i < len(file_contents) and file_contents[i] != '"':
                if file_contents[i] == '\n':
                    line += 1
                i += 1

            if i >= len(file_contents):
                print(
                    f"[line {string_line}] Error: Unterminated string.", file=sys.stderr)
                has_error = True
            else:
                i += 1
                lexeme = file_contents[start:i]
                literal = file_contents[start+1:i-1]
                tokens.append(Token("STRING", lexeme, literal, string_line))
        elif char.isdigit():
            # Number literal
            start = i
            while i < len(file_contents) and file_contents[i].isdigit():
                i += 1

            if i < len(file_contents) and file_contents[i] == '.' and i + 1 < len(file_contents) and file_contents[i + 1].isdigit():
                i += 1  # consume the '.'
                while i < len(file_contents) and file_contents[i].isdigit():
                    i += 1

            lexeme = file_contents[start:i]
            literal = float(lexeme)
            tokens.append(Token("NUMBER", lexeme, literal, line))
        elif char.isalpha() or char == '_':
            # Identifier or keyword
            start = i
            while i < len(file_contents) and (file_contents[i].isalnum() or file_contents[i] == '_'):
                i += 1

            lexeme = file_contents[start:i]

            if lexeme in keywords:
                tokens.append(Token(keywords[lexeme], lexeme, None, line))
            else:
                tokens.append(Token("IDENTIFIER", lexeme, None, line))
        elif char == '\n':
            line += 1
            i += 1
        elif char in ' \t\r':
            i += 1
        else:
            print(
                f"[line {line}] Error: Unexpected character: {char}", file=sys.stderr)
            has_error = True
            i += 1

    tokens.append(Token("EOF", "", None, line))
    return tokens, has_error


def main():
    """Main entry point for the interpreter."""
    if len(sys.argv) < 3:
        print("Usage: ./your_program.sh <command> <filename>", file=sys.stderr)
        exit(1)

    command = sys.argv[1]
    filename = sys.argv[2]

    if command not in ["tokenize", "parse"]:
        print(f"Unknown command: {command}", file=sys.stderr)
        exit(1)

    with open(filename) as file:
        file_contents = file.read()

    if command == "tokenize":
        tokens, has_error = tokenize(file_contents)

        for token in tokens:
            if token.literal is None:
                print(f"{token.type} {token.lexeme} null")
            else:
                print(f"{token.type} {token.lexeme} {token.literal}")

        # Exit with code 65 if there were errors
        if has_error:
            exit(65)

    elif command == "parse":
        tokens, has_error = tokenize(file_contents)

        if has_error:
            exit(65)

        parser = Parser(tokens)
        expr = parser.parse()

        if expr is not None:
            printer = AstPrinter()
            print(printer.print(expr))
        else:
            exit(65)


if __name__ == "__main__":
    main()
