"""Callable classes for the Lox interpreter."""

import time
from .environment import Environment
from .exceptions import Return, LoxRuntimeError


class LoxCallable:
    """Base class for callable objects."""

    def call(self, interpreter, arguments):
        """Execute the callable."""
        raise NotImplementedError()

    def arity(self):
        """Return the number of arguments expected."""
        raise NotImplementedError()


class ClockNative(LoxCallable):
    """Native clock function."""

    def call(self, interpreter, arguments):
        return time.time()

    def arity(self):
        return 0

    def __str__(self):
        return "<native fn>"


class BoundMethod(LoxCallable):
    """A method bound to an instance."""

    def __init__(self, instance, method):
        self.instance = instance
        self.method = method

    def call(self, interpreter, arguments):
        # Create environment for "this" (matches class scope during resolution)
        this_environment = Environment(self.method.closure)
        this_environment.define("this", self.instance)

        # Create environment for method parameters (nested inside this_environment)
        environment = Environment(this_environment)

        # Bind parameters to arguments
        for i, param in enumerate(self.method.declaration.params):
            environment.define(param.lexeme, arguments[i])

        # Execute method body
        try:
            interpreter.execute_block(
                self.method.declaration.body, environment)
        except Return as return_value:
            # Initializers always return 'this', even with explicit return
            if self.method.is_initializer:
                return self.instance
            return return_value.value

        # Initializers return 'this' by default
        if self.method.is_initializer:
            return self.instance
        return None

    def arity(self):
        return self.method.arity()

    def __str__(self):
        return f"<fn {self.method.declaration.name.lexeme}>"


class LoxFunction(LoxCallable):
    """User-defined Lox function."""

    def __init__(self, declaration, closure, is_initializer=False):
        self.declaration = declaration
        self.closure = closure
        self.is_initializer = is_initializer

    def bind(self, instance):
        """Bind this function to an instance."""
        return BoundMethod(instance, self)

    def call(self, interpreter, arguments):
        environment = Environment(self.closure)

        for i, param in enumerate(self.declaration.params):
            environment.define(param.lexeme, arguments[i])

        try:
            interpreter.execute_block(self.declaration.body, environment)
        except Return as return_value:
            return return_value.value

        return None

    def arity(self):
        return len(self.declaration.params)

    def __str__(self):
        return f"<fn {self.declaration.name.lexeme}>"


class LoxClass(LoxCallable):
    """Lox class."""

    def __init__(self, name, superclass, methods):
        self.name = name
        self.superclass = superclass
        self.methods = methods

    def find_method(self, name):
        """Find a method by name."""
        if name in self.methods:
            return self.methods[name]
        
        if self.superclass is not None:
            return self.superclass.find_method(name)
        
        return None

    def call(self, interpreter, arguments):
        """Instantiate the class."""
        instance = LoxInstance(self)
        
        initializer = self.find_method("init")
        if initializer is not None:
            initializer.bind(instance).call(interpreter, arguments)
        
        return instance

    def arity(self):
        """Return the arity of the init method if it exists."""
        initializer = self.find_method("init")
        if initializer is not None:
            return initializer.arity()
        return 0

    def __str__(self):
        return self.name


class LoxInstance:
    """Lox class instance."""

    def __init__(self, klass):
        self.klass = klass
        self.fields = {}

    def get(self, name):
        """Get a property from the instance."""
        if name.lexeme in self.fields:
            return self.fields[name.lexeme]


        method = self.klass.find_method(name.lexeme)
        if method is not None:
            return method.bind(self)

        raise LoxRuntimeError(name, f"Undefined property '{name.lexeme}'.")

    def set(self, name, value):
        """Set a property on the instance."""
        self.fields[name.lexeme] = value

    def __str__(self):
        return f"{self.klass.name} instance"
