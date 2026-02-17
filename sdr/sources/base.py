"""Abstract base class for message sources."""

from abc import ABC, abstractmethod

from sdr.models import InboundMessage


class MessageSource(ABC):
    """Base class that all message sources (Gmail, LinkedIn, etc.) must implement."""

    @abstractmethod
    def poll(self) -> list[InboundMessage]:
        """Fetch new messages since the last poll.

        Returns:
            A list of new inbound messages normalized to the InboundMessage format.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the source is healthy and ready to poll.

        Returns:
            True if the source can be reached and authenticated, False otherwise.
        """
        ...
