"""AST Printer for the Lox interpreter."""

from .expr import Literal, Binary, Unary, Grouping # pylint: disable=relative-beyond-top-level


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
