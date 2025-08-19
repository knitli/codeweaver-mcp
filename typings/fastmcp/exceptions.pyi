



class FastMCPError(Exception):
    ...


class ValidationError(FastMCPError):
    ...


class ResourceError(FastMCPError):
    ...


class ToolError(FastMCPError):
    ...


class PromptError(FastMCPError):
    ...


class InvalidSignature(Exception):
    ...


class ClientError(Exception):
    ...


class NotFoundError(Exception):
    ...


class DisabledError(Exception):
    ...

