try:
    import wandb
except ImportError:
    print("You need to install wandb!")


from typing import Union, Optional
from emmental.logging.log_writer import LogWriter
from emmental.meta import Meta


class WandbWriter(LogWriter):
    """A class for logging to Tensorboard during training process."""

    def __init__(
        self, project: Optional[str] = None, name: Optional[str] = None, **wandb_kwargs
    ) -> None:
        """Initialize TensorBoardWriter."""
        super().__init__()
        self.experiment = (
            wandb.init(
                name=name,
                project=project,
                dir=Meta.log_path,
                config=Meta.config,
                **wandb_kwargs
            )
            if wandb.run is None
            else wandb.run
        )

    def add_scalar(
        self, name: str, value: Union[float, int], step: Union[float, int]
    ) -> None:
        """Log a scalar variable.

        Args:
          name: The name of the scalar.
          value: The value of the scalar.
          step: The current step.
        """
        info = {name: value}
        self.experiment.log(info, step=int(step))

    def write_log(self, log_filename: str = "log.json") -> None:
        """Dump the log to file.

        Args:
          log_filename: The log filename, defaults to "log.json".
        """
        pass

    def close(self) -> None:
        """Close the tensorboard writer."""
        self.experiment.join()
