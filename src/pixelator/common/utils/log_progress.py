"""Logging utility for reporting progress updates in long-running operations.

Copyright © 2026 Pixelgen Technologies AB.
"""

import time
from logging import Logger


class LogProgress:
    """Utility for logging progress updates at a controlled interval.

    This class helps track and log the progress of processing items in a loop or batch job.
    It periodically writes progress messages to the provided logger, including elapsed time,
    items processed, and processing speed.
    """

    def __init__(
        self,
        logger: Logger,
        min_update_intervall_seconds: float = 60.0,
        min_update_intervall_items: int = 1,
        item_name: str = "read",
    ):
        """Initialize a LogProgress instance.

        Args:
            logger: Logger object to which progress messages will be written.
            min_update_intervall_seconds: Minimum seconds between log updates. Defaults to 60.
            min_update_intervall_items: Minimum number of items between log updates. Defaults to 1.
            item_name: Name of the item being processed. Defaults to "read".
        """
        self.logger = logger
        self.min_update_intervall_seconds = min_update_intervall_seconds
        self.min_update_intervall_items = min_update_intervall_items
        self.item_name = item_name
        self.n_items = 0
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_n_items = 0

    def update(self, increment_items: int, final_update: bool = False):
        """Update the progress and log a message if enough time has passed or if final update.

        Args:
            increment_items: Number of items processed since the last update.
            final_update: If True, forces a final log update regardless of interval. Defaults to False.
        """
        self.n_items += increment_items
        now = time.time()
        time_delta = now - (self.start_time if final_update else self.last_time)
        delta_items = self.n_items if final_update else self.n_items - self.last_n_items
        if delta_items < self.min_update_intervall_items:
            return
        if not final_update:
            if time_delta < self.min_update_intervall_seconds:
                return

        seconds_since_start = now - self.start_time
        hours = int(seconds_since_start) // 3600
        minutes = (int(seconds_since_start) - hours * 3600) // 60
        seconds = int(seconds_since_start) % 60
        items_per_second = delta_items / time_delta
        seconds_per_item = time_delta / delta_items

        self.logger.info(
            ("Progress: " if not final_update else "FINISHED: ")
            + f"{hours:02d}:{minutes:02d}:{seconds:02d} "
            + f"{self.n_items:13,d} {self.item_name}s "
            + f"@ {seconds_per_item * 1e6:5.1F} us/{self.item_name}; "
            + f"{items_per_second * 60 / 1e6:6.2F} M {self.item_name}s/minute"
        )
        self.last_time = now
        self.last_n_items = self.n_items

    def close(self):
        """Log a final progress update and close the progress logger.

        This should be called at the end of processing to ensure the last update is logged.
        """
        self.update(0, final_update=True)
