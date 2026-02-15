"""Environment class for variable storage in the Lox interpreter."""


class Environment:
    """Environment for storing variables."""

    def __init__(self, enclosing=None):
        self.values = {}
        self.enclosing = enclosing

    def define(self, name, value):
        """Define a new variable."""
        self.values[name] = value

    def get(self, name):
        """Get the value of a variable."""
        if name in self.values:
            return self.values[name]

        if self.enclosing is not None:
            return self.enclosing.get(name)

        raise RuntimeError(f"Undefined variable '{name}'.")

    def get_at(self, distance, name):
        """Get a variable at a specific distance in the environment chain."""
        env = self.ancestor(distance)
        if env is None:
            raise RuntimeError(f"Environment not found at distance {distance}")
        return env.values[name]

    def ancestor(self, distance):
        """Get the environment at a specific distance."""
        environment = self
        for _ in range(distance):
            if environment is None:
                return None
            environment = environment.enclosing
        return environment

    def assign(self, name, value):
        """Assign a value to an existing variable."""
        if name in self.values:
            self.values[name] = value
            return

        if self.enclosing is not None:
            self.enclosing.assign(name, value)
            return

        raise RuntimeError(f"Undefined variable '{name}'.")

    def assign_at(self, distance, name, value):
        """Assign a value to a variable at a specific distance."""
        env = self.ancestor(distance)
        if env is None:
            raise RuntimeError(f"Environment not found at distance {distance}")
        env.values[name] = value
