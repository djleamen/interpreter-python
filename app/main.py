"""
Code Interpreter - A simple interpreter in Python
From CodeCrafters.io build-your-own-interpreter (Python)
"""

import sys


def main():
    """Main entry point for the interpreter."""
    if len(sys.argv) < 3:
        print("Usage: ./your_program.sh tokenize <filename>", file=sys.stderr)
        exit(1)

    command = sys.argv[1]
    filename = sys.argv[2]

    if command != "tokenize":
        print(f"Unknown command: {command}", file=sys.stderr)
        exit(1)

    with open(filename) as file:
        file_contents = file.read()

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)

    has_error = False

    # Tokenize the file contents
    for char in file_contents:
        if char == '(':
            print("LEFT_PAREN ( null")
        elif char == ')':
            print("RIGHT_PAREN ) null")
        elif char == '{':
            print("LEFT_BRACE { null")
        elif char == '}':
            print("RIGHT_BRACE } null")
        elif char == ',':
            print("COMMA , null")
        elif char == '.':
            print("DOT . null")
        elif char == '-':
            print("MINUS - null")
        elif char == '+':
            print("PLUS + null")
        elif char == ';':
            print("SEMICOLON ; null")
        elif char == '*':
            print("STAR * null")
        elif char in ' \t\r\n':
            pass
        else:
            print(f"[line 1] Error: Unexpected character: {char}", file=sys.stderr)
            has_error = True

    print("EOF  null")

    # Exit with code 65 if there were errors
    if has_error:
        exit(65)

if __name__ == "__main__":
    main()
