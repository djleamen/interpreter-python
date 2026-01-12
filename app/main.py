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
    i = 0
    line = 1

    # Tokenize the file contents
    while i < len(file_contents):
        char = file_contents[i]

        if char == '(':
            print("LEFT_PAREN ( null")
            i += 1
        elif char == ')':
            print("RIGHT_PAREN ) null")
            i += 1
        elif char == '{':
            print("LEFT_BRACE { null")
            i += 1
        elif char == '}':
            print("RIGHT_BRACE } null")
            i += 1
        elif char == ',':
            print("COMMA , null")
            i += 1
        elif char == '.':
            print("DOT . null")
            i += 1
        elif char == '-':
            print("MINUS - null")
            i += 1
        elif char == '+':
            print("PLUS + null")
            i += 1
        elif char == ';':
            print("SEMICOLON ; null")
            i += 1
        elif char == '*':
            print("STAR * null")
            i += 1
        elif char == '/':
            if i + 1 < len(file_contents) and file_contents[i + 1] == '/':
                i += 2
                while i < len(file_contents) and file_contents[i] != '\n':
                    i += 1
            else:
                print("SLASH / null")
                i += 1
        elif char == '=':
            if i + 1 < len(file_contents) and file_contents[i + 1] == '=':
                print("EQUAL_EQUAL == null")
                i += 2
            else:
                print("EQUAL = null")
                i += 1
        elif char == '!':
            if i + 1 < len(file_contents) and file_contents[i + 1] == '=':
                print("BANG_EQUAL != null")
                i += 2
            else:
                print("BANG ! null")
                i += 1
        elif char == '<':
            if i + 1 < len(file_contents) and file_contents[i + 1] == '=':
                print("LESS_EQUAL <= null")
                i += 2
            else:
                print("LESS < null")
                i += 1
        elif char == '>':
            if i + 1 < len(file_contents) and file_contents[i + 1] == '=':
                print("GREATER_EQUAL >= null")
                i += 2
            else:
                print("GREATER > null")
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
                print(f"STRING {lexeme} {literal}")
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
            # Format the number literal
            if '.' in lexeme:
                literal = float(lexeme)
            else:
                literal = float(lexeme)  # Still use float for consistency
            print(f"NUMBER {lexeme} {literal}")
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

    print("EOF  null")

    # Exit with code 65 if there were errors
    if has_error:
        exit(65)


if __name__ == "__main__":
    main()
