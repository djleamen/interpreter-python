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


class Variable(Expr):
    """Variable expression."""

    def __init__(self, name):
        self.name = name


class Assign(Expr):
    """Assignment expression."""

    def __init__(self, name, value):
        self.name = name
        self.value = value


class Logical(Expr):
    """Logical expression (and/or)."""

    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right


# Statement classes
class Stmt:
    """Base class for statements."""
    pass


class PrintStmt(Stmt):
    """Print statement."""

    def __init__(self, expression):
        self.expression = expression


class ExpressionStmt(Stmt):
    """Expression statement."""

    def __init__(self, expression):
        self.expression = expression


class VarStmt(Stmt):
    """Variable declaration statement."""

    def __init__(self, name, initializer):
        self.name = name
        self.initializer = initializer


class BlockStmt(Stmt):
    """Block statement."""

    def __init__(self, statements):
        self.statements = statements


class IfStmt(Stmt):
    """If statement."""

    def __init__(self, condition, then_branch, else_branch):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch


class WhileStmt(Stmt):
    """While statement."""

    def __init__(self, condition, body):
        self.condition = condition
        self.body = body


class ParseException(Exception):
    """Exception raised during parsing."""
    pass


class LoxRuntimeError(Exception):
    """Exception raised during runtime evaluation."""

    def __init__(self, token, message):
        self.token = token
        self.message = message
        super().__init__(message)


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


class Environment:
    """Environment for storing variables."""

    def __init__(self, enclosing=None):
        self.values = {}
        self.enclosing = enclosing

    def define(self, name, value):
        """Define a new variable."""
        self.values[name] = value

    def get(self, name):
        """Get the value of a variable."""
        if name in self.values:
            return self.values[name]

        if self.enclosing is not None:
            return self.enclosing.get(name)

        raise RuntimeError(f"Undefined variable '{name}'.")

    def assign(self, name, value):
        """Assign a value to an existing variable."""
        if name in self.values:
            self.values[name] = value
            return

        if self.enclosing is not None:
            self.enclosing.assign(name, value)
            return

        raise RuntimeError(f"Undefined variable '{name}'.")


class Interpreter:
    """Class to evaluate expressions."""

    def __init__(self):
        self.environment = Environment()

    def execute(self, stmt):
        """Execute a statement."""
        if isinstance(stmt, PrintStmt):
            value = self.evaluate(stmt.expression)
            print(self.stringify(value))
        elif isinstance(stmt, ExpressionStmt):
            self.evaluate(stmt.expression)
        elif isinstance(stmt, VarStmt):
            value = None
            if stmt.initializer is not None:
                value = self.evaluate(stmt.initializer)
            self.environment.define(stmt.name.lexeme, value)
        elif isinstance(stmt, BlockStmt):
            self.execute_block(stmt.statements, Environment(self.environment))
        elif isinstance(stmt, IfStmt):
            if self.is_truthy(self.evaluate(stmt.condition)):
                self.execute(stmt.then_branch)
            elif stmt.else_branch is not None:
                self.execute(stmt.else_branch)
        elif isinstance(stmt, WhileStmt):
            while self.is_truthy(self.evaluate(stmt.condition)):
                self.execute(stmt.body)

    def execute_block(self, statements, environment):
        """Execute a block of statements in the given environment."""
        previous = self.environment
        try:
            self.environment = environment
            for statement in statements:
                self.execute(statement)
        finally:
            self.environment = previous

    def evaluate(self, expr):
        """Evaluate an expression and return its value."""
        if isinstance(expr, Literal):
            return expr.value
        elif isinstance(expr, Variable):
            try:
                return self.environment.get(expr.name.lexeme)
            except RuntimeError as e:
                raise LoxRuntimeError(expr.name, str(e)) from e
        elif isinstance(expr, Assign):
            value = self.evaluate(expr.value)
            try:
                self.environment.assign(expr.name.lexeme, value)
            except RuntimeError as e:
                raise LoxRuntimeError(expr.name, str(e)) from e
            return value
        elif isinstance(expr, Grouping):
            return self.evaluate(expr.expression)
        elif isinstance(expr, Logical):
            left = self.evaluate(expr.left)

            if expr.operator.type == "OR":
                if self.is_truthy(left):
                    return left
            else:  # AND
                if not self.is_truthy(left):
                    return left

            return self.evaluate(expr.right)
        elif isinstance(expr, Unary):
            right = self.evaluate(expr.right)
            if expr.operator.type == "MINUS":
                self.check_number_operand(expr.operator, right)
                return -right
            elif expr.operator.type == "BANG":
                return not self.is_truthy(right)
        elif isinstance(expr, Binary):
            left = self.evaluate(expr.left)
            right = self.evaluate(expr.right)

            if expr.operator.type == "MINUS":
                self.check_number_operands(expr.operator, left, right)
                return left - right
            elif expr.operator.type == "PLUS":
                if isinstance(left, float) and isinstance(right, float):
                    return left + right
                if isinstance(left, str) and isinstance(right, str):
                    return left + right
                raise LoxRuntimeError(
                    expr.operator, "Operands must be two numbers or two strings.")
            elif expr.operator.type == "SLASH":
                self.check_number_operands(expr.operator, left, right)
                return left / right
            elif expr.operator.type == "STAR":
                self.check_number_operands(expr.operator, left, right)
                return left * right
            elif expr.operator.type == "GREATER":
                self.check_number_operands(expr.operator, left, right)
                return left > right
            elif expr.operator.type == "GREATER_EQUAL":
                self.check_number_operands(expr.operator, left, right)
                return left >= right
            elif expr.operator.type == "LESS":
                self.check_number_operands(expr.operator, left, right)
                return left < right
            elif expr.operator.type == "LESS_EQUAL":
                self.check_number_operands(expr.operator, left, right)
                return left <= right
            elif expr.operator.type == "EQUAL_EQUAL":
                return self.is_equal(left, right)
            elif expr.operator.type == "BANG_EQUAL":
                return not self.is_equal(left, right)
        return None

    def is_truthy(self, value):
        """Determine if a value is truthy in Lox."""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        return True

    def is_equal(self, a, b):
        """Check if two values are equal."""
        if a is None and b is None:
            return True
        if a is None:
            return False
        return a == b

    def stringify(self, value):
        """Convert a value to its string representation."""
        if value is None:
            return "nil"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            text = str(value)
            # Remove .0 suffix for whole numbers
            if text.endswith(".0"):
                text = text[:-2]
            return text
        return str(value)

    def check_number_operand(self, operator, operand):
        """Check if operand is a number, raise LoxRuntimeError if not."""
        if isinstance(operand, float):
            return
        raise LoxRuntimeError(operator, "Operand must be a number.")

    def check_number_operands(self, operator, left, right):
        """Check if both operands are numbers, raise LoxRuntimeError if not."""
        if isinstance(left, float) and isinstance(right, float):
            return
        raise LoxRuntimeError(operator, "Operands must be numbers.")


class Parser:
    """A simple recursive descent parser."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0
        self.had_error = False

    def parse(self):
        """Parse the tokens and return the expression."""
        try:
            return self.expression()
        except ParseException:
            return None

    def parse_statements(self):
        """Parse multiple statements and return a list."""
        statements = []
        while not self.is_at_end():
            try:
                stmt = self.declaration()
                if stmt is not None:
                    statements.append(stmt)
            except ParseException:
                self.synchronize()
        return statements

    def declaration(self):
        """Parse a declaration."""
        if self.match("VAR"):
            return self.var_declaration()
        return self.statement()

    def var_declaration(self):
        """Parse a variable declaration."""
        name = self.consume("IDENTIFIER", "Expect variable name.")

        initializer = None
        if self.match("EQUAL"):
            initializer = self.expression()

        self.consume("SEMICOLON", "Expect ';' after variable declaration.")
        return VarStmt(name, initializer)

    def statement(self):
        """Parse a single statement."""
        if self.match("PRINT"):
            return self.print_statement()
        if self.match("LEFT_BRACE"):
            return BlockStmt(self.block())
        if self.match("IF"):
            return self.if_statement()
        if self.match("WHILE"):
            return self.while_statement()
        if self.match("FOR"):
            return self.for_statement()
        return self.expression_statement()

    def block(self):
        """Parse a block of statements."""
        statements = []

        while not self.check("RIGHT_BRACE") and not self.is_at_end():
            statements.append(self.declaration())

        self.consume("RIGHT_BRACE", "Expect '}'.")
        return statements

    def print_statement(self):
        """Parse a print statement."""
        expr = self.expression()
        self.consume("SEMICOLON", "Expect ';' after value.")
        return PrintStmt(expr)

    def if_statement(self):
        """Parse an if statement."""
        self.consume("LEFT_PAREN", "Expect '(' after 'if'.")
        condition = self.expression()
        self.consume("RIGHT_PAREN", "Expect ')' after if condition.")

        then_branch = self.statement()
        else_branch = None
        if self.match("ELSE"):
            else_branch = self.statement()

        return IfStmt(condition, then_branch, else_branch)

    def while_statement(self):
        """Parse a while statement."""
        self.consume("LEFT_PAREN", "Expect '(' after 'while'.")
        condition = self.expression()
        self.consume("RIGHT_PAREN", "Expect ')' after condition.")
        body = self.statement()

        return WhileStmt(condition, body)

    def for_statement(self):
        """Parse a for statement (desugars to while)."""
        self.consume("LEFT_PAREN", "Expect '(' after 'for'.")

        # Initializer
        initializer = None
        if self.match("SEMICOLON"):
            initializer = None
        elif self.match("VAR"):
            initializer = self.var_declaration()
        else:
            initializer = self.expression_statement()

        # Condition
        condition = None
        if not self.check("SEMICOLON"):
            condition = self.expression()
        self.consume("SEMICOLON", "Expect ';' after loop condition.")

        # Increment
        increment = None
        if not self.check("RIGHT_PAREN"):
            increment = self.expression()
        self.consume("RIGHT_PAREN", "Expect ')' after for clauses.")

        body = self.statement()

        if increment is not None:
            body = BlockStmt([body, ExpressionStmt(increment)])

        if condition is None:
            condition = Literal(True)
        body = WhileStmt(condition, body)

        if initializer is not None:
            body = BlockStmt([initializer, body])

        return body

    def expression_statement(self):
        """Parse an expression statement."""
        expr = self.expression()
        self.consume("SEMICOLON", "Expect ';' after expression.")
        return ExpressionStmt(expr)

    def synchronize(self):
        """Synchronize after a parse error."""
        self.advance()
        while not self.is_at_end():
            if self.previous().type == "SEMICOLON":
                return
            if self.peek().type in ["CLASS", "FUN", "VAR", "FOR", "IF", "WHILE", "PRINT", "RETURN"]:
                return
            self.advance()

    def error(self, token, message):
        """Report an error at the given token."""
        if token.type == "EOF":
            print(f"[line {token.line}] Error at end: {message}",
                  file=sys.stderr)
        else:
            print(
                f"[line {token.line}] Error at '{token.lexeme}': {message}", file=sys.stderr)
        self.had_error = True
        raise ParseException(message)

    def expression(self):
        """Parse an expression."""
        return self.assignment()

    def assignment(self):
        """Parse an assignment expression."""
        expr = self.or_expression()

        if self.match("EQUAL"):
            equals = self.previous()
            value = self.assignment()  # Right-associative

            if isinstance(expr, Variable):
                name = expr.name
                return Assign(name, value)

            self.error(equals, "Invalid assignment target.")

        return expr

    def or_expression(self):
        """Parse logical or expression."""
        expr = self.and_expression()

        while self.match("OR"):
            operator = self.previous()
            right = self.and_expression()
            expr = Logical(expr, operator, right)

        return expr

    def and_expression(self):
        """Parse logical and expression."""
        expr = self.equality()

        while self.match("AND"):
            operator = self.previous()
            right = self.equality()
            expr = Logical(expr, operator, right)

        return expr

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
        if self.match("IDENTIFIER"):
            return Variable(self.previous())

        self.error(self.peek(), "Expect expression.")

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
        self.error(self.peek(), message)


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

    if command not in ["tokenize", "parse", "evaluate", "run"]:
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

        if expr is not None and not parser.had_error:
            printer = AstPrinter()
            print(printer.print(expr))
        else:
            exit(65)

    elif command == "evaluate":
        tokens, has_error = tokenize(file_contents)

        if has_error:
            exit(65)

        parser = Parser(tokens)
        expr = parser.parse()

        if expr is None or parser.had_error:
            exit(65)

        interpreter = Interpreter()
        try:
            value = interpreter.evaluate(expr)
            print(interpreter.stringify(value))
        except LoxRuntimeError as e:
            print(e.message, file=sys.stderr)
            print(f"[line {e.token.line}]", file=sys.stderr)
            exit(70)

    elif command == "run":
        tokens, has_error = tokenize(file_contents)

        if has_error:
            exit(65)

        parser = Parser(tokens)
        statements = parser.parse_statements()

        if parser.had_error:
            exit(65)

        interpreter = Interpreter()
        try:
            for stmt in statements:
                interpreter.execute(stmt)
        except LoxRuntimeError as e:
            print(e.message, file=sys.stderr)
            print(f"[line {e.token.line}]", file=sys.stderr)
            exit(70)


if __name__ == "__main__":
    main()
