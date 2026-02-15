"""Interpreter for the Lox language."""

from .environment import Environment
from .callable import ClockNative, LoxCallable, LoxFunction, LoxClass, LoxInstance
from .exceptions import LoxRuntimeError, Return
from .stmt import (
    PrintStmt, ExpressionStmt, VarStmt, BlockStmt, IfStmt,
    WhileStmt, FunStmt, ClassStmt, ReturnStmt
)
from .expr import (
    Literal, Super, This, Variable, Assign, Grouping, Logical,
    Unary, Binary, Call, Get, Set
)


class Interpreter:
    """Class to evaluate expressions."""

    def __init__(self):
        self.globals = Environment()
        self.environment = self.globals
        self.globals.define("clock", ClockNative())
        self.locals = {}  # Maps expressions to their resolved depths

    def resolve(self, expr, depth):
        """Store the resolved depth for an expression."""
        self.locals[id(expr)] = depth

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
        elif isinstance(stmt, FunStmt):
            function = LoxFunction(stmt, self.environment)
            self.environment.define(stmt.name.lexeme, function)
        elif isinstance(stmt, ClassStmt):
            superclass = None
            if stmt.superclass is not None:
                superclass = self.evaluate(stmt.superclass)
                if not isinstance(superclass, LoxClass):
                    raise LoxRuntimeError(stmt.superclass.name,
                                         "Superclass must be a class.")
            
            self.environment.define(stmt.name.lexeme, None)

            if stmt.superclass is not None:
                self.environment = Environment(self.environment)
                self.environment.define("super", superclass)

            # Create methods - they capture the current environment
            methods = {}
            for method in stmt.methods:
                is_initializer = method.name.lexeme == "init"
                function = LoxFunction(method, self.environment, is_initializer)
                methods[method.name.lexeme] = function

            klass = LoxClass(stmt.name.lexeme, superclass, methods)
            
            if stmt.superclass is not None:
                self.environment = self.environment.enclosing
            
            self.environment.values[stmt.name.lexeme] = klass
        elif isinstance(stmt, ReturnStmt):
            value = None
            if stmt.value is not None:
                value = self.evaluate(stmt.value)
            raise Return(value)

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
        elif isinstance(expr, Super):
            distance = self.locals.get(id(expr))
            superclass = self.environment.get_at(distance, "super")
            
            # "this" is always one level nearer than "super"
            obj = self.environment.get_at(distance - 1, "this")
            
            method = superclass.find_method(expr.method.lexeme)
            
            if method is None:
                raise LoxRuntimeError(expr.method,
                                    f"Undefined property '{expr.method.lexeme}'.")
            
            return method.bind(obj)
        elif isinstance(expr, This):
            return self.lookup_variable(expr.keyword, expr)
        elif isinstance(expr, Variable):
            return self.lookup_variable(expr.name, expr)
        elif isinstance(expr, Assign):
            value = self.evaluate(expr.value)
            distance = self.locals.get(id(expr))
            if distance is not None:
                self.environment.assign_at(distance, expr.name.lexeme, value)
            else:
                self.globals.assign(expr.name.lexeme, value)
            return value
        elif isinstance(expr, Grouping):
            return self.evaluate(expr.expression)
        elif isinstance(expr, Logical):
            left = self.evaluate(expr.left)

            if expr.operator.type == "OR":
                if self.is_truthy(left):
                    return left
            else:
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
        elif isinstance(expr, Call):
            callee = self.evaluate(expr.callee)

            arguments = []
            for argument in expr.arguments:
                arguments.append(self.evaluate(argument))

            if not isinstance(callee, LoxCallable):
                raise LoxRuntimeError(
                    expr.paren, "Can only call functions and classes.")

            function = callee
            if len(arguments) != function.arity():
                raise LoxRuntimeError(
                    expr.paren,
                    f"Expected {function.arity()} arguments but got {len(arguments)}.")

            return function.call(self, arguments)
        elif isinstance(expr, Get):
            obj = self.evaluate(expr.obj)
            if isinstance(obj, LoxInstance):
                return obj.get(expr.name)
            raise LoxRuntimeError(expr.name, "Only instances have properties.")
        elif isinstance(expr, Set):
            obj = self.evaluate(expr.obj)

            if not isinstance(obj, LoxInstance):
                raise LoxRuntimeError(expr.name, "Only instances have fields.")

            value = self.evaluate(expr.value)
            obj.set(expr.name, value)
            return value
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

    def lookup_variable(self, name, expr):
        """Look up a variable using resolved depth if available."""
        distance = self.locals.get(id(expr))
        if distance is not None:
            return self.environment.get_at(distance, name.lexeme)
        else:
            try:
                return self.globals.get(name.lexeme)
            except RuntimeError as e:
                raise LoxRuntimeError(name, str(e)) from e
