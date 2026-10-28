class AdapterError(ValueError):
    """Fail closed with a stable, machine-readable reason and human detail."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = str(detail)
        super().__init__("{}: {}".format(code, detail))
