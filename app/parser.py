"""Parser for the Lox language."""

import sys
from .token import Token
from .exceptions import ParseException
from .expr import (
    Expr, Literal, Binary, Unary, Grouping, Variable, Assign,
    Logical, Call, Get, Set, This, Super
)
from .stmt import (
    PrintStmt, ExpressionStmt, VarStmt, BlockStmt, IfStmt,
    WhileStmt, FunStmt, ClassStmt, ReturnStmt
)


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
        if self.match("CLASS"):
            return self.class_declaration()
        if self.match("FUN"):
            return self.function("function")
        if self.match("VAR"):
            return self.var_declaration()
        return self.statement()

    def class_declaration(self):
        """Parse a class declaration."""
        name = self.consume("IDENTIFIER", "Expect class name.")
        
        superclass = None
        if self.match("LESS"):
            self.consume("IDENTIFIER", "Expect superclass name.")
            superclass = Variable(self.previous())
        
        self.consume("LEFT_BRACE", "Expect '{' before class body.")

        methods = []
        while not self.check("RIGHT_BRACE") and not self.is_at_end():
            methods.append(self.function("method"))

        self.consume("RIGHT_BRACE", "Expect '}' after class body.")
        return ClassStmt(name, superclass, methods)

    def function(self, kind):
        """Parse a function declaration."""
        name = self.consume("IDENTIFIER", f"Expect {kind} name.")
        self.consume("LEFT_PAREN", f"Expect '(' after {kind} name.")

        parameters = []
        if not self.check("RIGHT_PAREN"):
            while True:
                if len(parameters) >= 255:
                    self.error(
                        self.peek(), "Can't have more than 255 parameters.")

                parameters.append(self.consume(
                    "IDENTIFIER", "Expect parameter name."))

                if not self.match("COMMA"):
                    break

        self.consume("RIGHT_PAREN", "Expect ')' after parameters.")
        self.consume("LEFT_BRACE", f"Expect '{{' before {kind} body.")
        body = self.block()

        return FunStmt(name, parameters, body)

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
        if self.match("RETURN"):
            return self.return_statement()
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

    def return_statement(self):
        """Parse a return statement."""
        keyword = self.previous()
        value = None
        if not self.check("SEMICOLON"):
            value = self.expression()

        self.consume("SEMICOLON", "Expect ';' after return value.")
        return ReturnStmt(keyword, value)

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
            elif isinstance(expr, Get):
                return Set(expr.obj, expr.name, value)

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
        return self.call()

    def call(self):
        """Parse call expressions."""
        expr = self.primary()

        while True:
            if self.match("LEFT_PAREN"):
                expr = self.finish_call(expr)
            elif self.match("DOT"):
                name = self.consume(
                    "IDENTIFIER", "Expect property name after '.'.")
                expr = Get(expr, name)
            else:
                break

        return expr

    def finish_call(self, callee):
        """Finish parsing a call expression."""
        arguments = []

        if not self.check("RIGHT_PAREN"):
            while True:
                arguments.append(self.expression())
                if not self.match("COMMA"):
                    break

        paren = self.consume("RIGHT_PAREN", "Expect ')' after arguments.")
        return Call(callee, paren, arguments)

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
        if self.match("SUPER"):
            keyword = self.previous()
            self.consume("DOT", "Expect '.' after 'super'.")
            method = self.consume("IDENTIFIER", "Expect superclass method name.")
            return Super(keyword, method)
        if self.match("THIS"):
            return This(self.previous())
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
