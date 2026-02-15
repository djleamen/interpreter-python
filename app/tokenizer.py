"""Tokenizer for the Lox language."""

import sys
from .token import Token


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
