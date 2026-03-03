import torch
from beat_this.inference import File2Beats


class BeatThisTracker:
    def __init__(self, checkpoint_path: str = "final0", dbn: bool = False) -> None:
        self.checkpoint_path = checkpoint_path
        self.dbn = dbn
        self._tracker = None

    def track(self, audio_filename: str):
        if not hasattr(torch.amp, "is_autocast_available"):

            def is_autocast_available(device_type):
                del device_type
                return False

            torch.amp.is_autocast_available = is_autocast_available

        if self._tracker is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tracker = File2Beats(
                checkpoint_path=self.checkpoint_path,
                device=device,
                dbn=self.dbn
            )
        return self._tracker(audio_filename)
