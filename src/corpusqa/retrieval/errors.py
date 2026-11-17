"""The one error a corpus tool raises on purpose: the model asked for something that is not
there (a field, a value, a handle), and the message says what to do instead.

The agent's ToolBox shows its message to the model as `<tool_error>...</tool_error>`. Any
other exception is a bug: the model sees only its type, the traceback goes to the log.
So the message must be written for the model, from values the tool chose to show, and
must never carry exception text, paths or secrets.
"""


class ToolInputError(ValueError):
    """The model's arguments name something the index does not have; `message` says what
    to do instead (e.g. "no field 'statu' in <source>; did you mean 'status'?")."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
