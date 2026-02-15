"""
Code Interpreter - A simple interpreter in Python
From CodeCrafters.io build-your-own-interpreter (Python)
"""

import sys
from .tokenizer import tokenize
from .parser import Parser
from .ast_printer import AstPrinter
from .interpreter import Interpreter
from .resolver import Resolver
from .exceptions import LoxRuntimeError


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

        # Resolve all identifiers before execution
        resolver = Resolver(interpreter)
        resolver.resolve(statements)

        if resolver.had_error:
            exit(65)

        try:
            for stmt in statements:
                interpreter.execute(stmt)
        except LoxRuntimeError as e:
            print(e.message, file=sys.stderr)
            print(f"[line {e.token.line}]", file=sys.stderr)
            exit(70)


if __name__ == "__main__":
    main()
