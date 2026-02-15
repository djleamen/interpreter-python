"""Statement classes for the Lox interpreter."""


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


class FunStmt(Stmt):
    """Function declaration statement."""

    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body


class ClassStmt(Stmt):
    """Class declaration statement."""

    def __init__(self, name, superclass, methods):
        self.name = name
        self.superclass = superclass
        self.methods = methods


class ReturnStmt(Stmt):
    """Return statement."""

    def __init__(self, keyword, value):
        self.keyword = keyword
        self.value = value
