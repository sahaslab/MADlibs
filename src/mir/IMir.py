from abc import ABC, abstractmethod
import numpy as np

class IMir(ABC):
    @abstractmethod
    def process_audio(self, id: str, audio: np.ndarray, sr: int = 44100) -> None:
        """
        Process the audio via given music information retrieval method
        Store the processed data in an internal class representation for future analysis

        Args:
            - audio: the quantized audio

        Returns:
            - nothing (store processed audio in class)
        """

    @abstractmethod
    def plot_analysis(self) -> None:
        """
        Plot the MIR analysis over all songs

        Args:
            - none
        Returns:
            - nothing (plots)
        """

    @abstractmethod
    def get_single_analysis(self, id: str) -> np.ndarray:
        """
        Get the analysis for one song by id

        Args:
            - id: str the id for the stored song
        Returns:
            - a numpy array of the analysis data
        """

    @abstractmethod
    def get_analysis(self) -> dict[str, np.ndarray]:
        """
        Get the analysis for all songs

        Args:
            - none
        Returns:
            - a dictionary of id, analysis data pairs
        """