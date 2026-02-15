"""Resolver for compile-time identifier resolution in the Lox interpreter."""

import sys
from .stmt import (  # pylint: disable=relative-beyond-top-level
    Stmt, BlockStmt, VarStmt, FunStmt, ClassStmt, ExpressionStmt,
    IfStmt, PrintStmt, ReturnStmt, WhileStmt
)
from .expr import (  # pylint: disable=relative-beyond-top-level
    Expr, Super, This, Variable, Assign, Binary, Call, Get, Set,
    Grouping, Literal, Logical, Unary
)


class Resolver:
    """Resolver for compile-time identifier resolution."""

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.scopes = []  # Stack of scopes
        self.had_error = False
        self.current_function = None  # None, "function", or "initializer"
        self.current_class = None  # None, "class", or "subclass"

    def resolve(self, item):
        """Resolve a statement, expression, or list of statements."""
        if isinstance(item, list):
            for statement in item:
                self.resolve(statement)
        elif isinstance(item, Stmt):
            self.resolve_stmt(item)
        elif isinstance(item, Expr):
            self.resolve_expr(item)

    def resolve_stmt(self, stmt):
        """Resolve a statement."""
        if isinstance(stmt, BlockStmt):
            self.begin_scope()
            self.resolve(stmt.statements)
            self.end_scope()
        elif isinstance(stmt, VarStmt):
            self.declare(stmt.name)
            if stmt.initializer is not None:
                self.resolve(stmt.initializer)
            self.define(stmt.name)
        elif isinstance(stmt, FunStmt):
            self.declare(stmt.name)
            self.define(stmt.name)
            self.resolve_function(stmt, "function")
        elif isinstance(stmt, ClassStmt):
            enclosing_class = self.current_class
            self.current_class = "class"

            self.declare(stmt.name)
            self.define(stmt.name)

            if stmt.superclass is not None:
                # Check if class inherits from itself
                if stmt.name.lexeme == stmt.superclass.name.lexeme:
                    self.error(stmt.superclass.name,
                               "A class can't inherit from itself.")
                self.current_class = "subclass"
                self.resolve(stmt.superclass)

                # Create scope for "super"
                self.begin_scope()
                self.scopes[-1]["super"] = True

            self.begin_scope()
            self.scopes[-1]["this"] = True

            for method in stmt.methods:
                declaration = "initializer" if method.name.lexeme == "init" else "method"
                self.resolve_function(method, declaration)

            self.end_scope()

            if stmt.superclass is not None:
                self.end_scope()

            self.current_class = enclosing_class
        elif isinstance(stmt, ExpressionStmt):
            self.resolve(stmt.expression)
        elif isinstance(stmt, IfStmt):
            self.resolve(stmt.condition)
            self.resolve(stmt.then_branch)
            if stmt.else_branch is not None:
                self.resolve(stmt.else_branch)
        elif isinstance(stmt, PrintStmt):
            self.resolve(stmt.expression)
        elif isinstance(stmt, ReturnStmt):
            if self.current_function is None:
                self.error(stmt.keyword, "Can't return from top-level code.")
            if stmt.value is not None:
                if self.current_function == "initializer":
                    self.error(
                        stmt.keyword, "Can't return a value from an initializer.")
                self.resolve(stmt.value)
        elif isinstance(stmt, WhileStmt):
            self.resolve(stmt.condition)
            self.resolve(stmt.body)

    def resolve_expr(self, expr):
        """Resolve an expression."""
        if isinstance(expr, Super):
            if self.current_class is None:
                self.error(
                    expr.keyword, "Can't use 'super' outside of a class.")
            elif self.current_class != "subclass":
                self.error(
                    expr.keyword, "Can't use 'super' in a class with no superclass.")
            else:
                self.resolve_local(expr, expr.keyword)
        elif isinstance(expr, This):
            if self.current_class is None:
                self.error(
                    expr.keyword, "Can't use 'this' outside of a class.")
                return
            self.resolve_local(expr, expr.keyword)
        elif isinstance(expr, Variable):
            if self.scopes and self.scopes[-1].get(expr.name.lexeme) is False:
                self.error(
                    expr.name, "Can't read local variable in its own initializer.")
            self.resolve_local(expr, expr.name)
        elif isinstance(expr, Assign):
            self.resolve(expr.value)
            self.resolve_local(expr, expr.name)
        elif isinstance(expr, Binary):
            self.resolve(expr.left)
            self.resolve(expr.right)
        elif isinstance(expr, Call):
            self.resolve(expr.callee)
            for argument in expr.arguments:
                self.resolve(argument)
        elif isinstance(expr, Get):
            self.resolve(expr.obj)
        elif isinstance(expr, Set):
            self.resolve(expr.value)
            self.resolve(expr.obj)
        elif isinstance(expr, Grouping):
            self.resolve(expr.expression)
        elif isinstance(expr, Literal):
            pass
        elif isinstance(expr, Logical):
            self.resolve(expr.left)
            self.resolve(expr.right)
        elif isinstance(expr, Unary):
            self.resolve(expr.right)

    def resolve_function(self, function, function_type="function"):
        """Resolve a function declaration."""
        enclosing_function = self.current_function
        self.current_function = function_type

        self.begin_scope()
        for param in function.params:
            self.declare(param)
            self.define(param)
        self.resolve(function.body)
        self.end_scope()

        self.current_function = enclosing_function

    def begin_scope(self):
        """Begin a new scope."""
        self.scopes.append({})

    def end_scope(self):
        """End the current scope."""
        self.scopes.pop()

    def declare(self, name):
        """Declare a variable in the current scope."""
        if not self.scopes:
            return
        scope = self.scopes[-1]
        if name.lexeme in scope:
            self.error(name, "Already a variable with this name in this scope.")
        scope[name.lexeme] = False

    def define(self, name):
        """Define a variable in the current scope."""
        if not self.scopes:
            return
        self.scopes[-1][name.lexeme] = True

    def resolve_local(self, expr, name):
        """Resolve a local variable."""
        for i in range(len(self.scopes) - 1, -1, -1):
            if name.lexeme in self.scopes[i]:
                self.interpreter.resolve(expr, len(self.scopes) - 1 - i)
                return

    def error(self, token, message):
        """Report an error."""
        print(
            f"[line {token.line}] Error at '{token.lexeme}': {message}", file=sys.stderr)
        self.had_error = True
