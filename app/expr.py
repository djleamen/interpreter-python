"""Expression classes for the Lox interpreter."""


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


class Call(Expr):
    """Call expression."""

    def __init__(self, callee, paren, arguments):
        self.callee = callee
        self.paren = paren
        self.arguments = arguments


class Get(Expr):
    """Property get expression."""

    def __init__(self, obj, name):
        self.obj = obj
        self.name = name


class Set(Expr):
    """Property set expression."""

    def __init__(self, obj, name, value):
        self.obj = obj
        self.name = name
        self.value = value


class This(Expr):
    """This expression."""

    def __init__(self, keyword):
        self.keyword = keyword


class Super(Expr):
    """Super expression."""

    def __init__(self, keyword, method):
        self.keyword = keyword
        self.method = method
