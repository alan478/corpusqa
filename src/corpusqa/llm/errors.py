"""Errors of the model clients. Callers catch `LLMError` and degrade (e.g. keyword-only)."""


class LLMError(RuntimeError):
    """A model call could not be made or gave an unusable answer."""


class SpendCapReached(LLMError):
    """The recorded spend would pass its cap; no call was made."""


class EmbeddingError(LLMError):
    """An embedding request failed or its answer did not match the texts sent."""


class BadEmbeddingAnswer(EmbeddingError):
    """The API answered, but not with one usable vector per text sent. It was billed, and
    sending the same request again is not expected to help."""


class EmbedderUnavailable(EmbeddingError):
    """No embedder can be built (no API key); search must fall back to keywords only."""


class ChatError(LLMError):
    """A chat (text generation) call failed or gave an unusable answer."""


class ChatTimeout(ChatError):
    """The chat call did not answer within its time limit (retries included)."""


class BadChatAnswer(ChatError):
    """The API answered, but with nothing usable (no candidate, or a blocked one)."""


class ChatUnavailable(ChatError):
    """No chat model can be built (no API key)."""
