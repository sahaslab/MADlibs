import csv
import dataclasses
from abc import ABC
from abc import abstractmethod
from pathlib import Path


class PromptStrategy(ABC):
    @abstractmethod
    def get_prompts(self, n_prompts: int, seed: int | None = None) -> list[str]:
        """Generate n_prompts randomly following defined strategy."""
        raise NotImplementedError


@dataclasses.dataclass(kw_only=True)
class CorpusPromptStrategy(PromptStrategy):
    """
    Where we assume there is a corpus of captions available
    """
    seed: int = 0
    caption_corpus_filename: str = ""
    caption_column: int = 2
    caption_has_header: bool = True
    caption_delimiter: str = ","
    caption_filter_column: int | None = 3
    caption_filter_value: str | None = "True"

    def read_captions_from_corpus(self) -> list[str]:
        caption_path = Path(self.caption_corpus_filename)
        if caption_path.suffix.lower() in [".csv", ".tsv"]:
            return self.read_tabular_captions(caption_path)
        return self.read_text_captions(caption_path)

    def read_tabular_captions(self, caption_path: Path) -> list[str]:
        delimiter = self.caption_delimiter
        if caption_path.suffix.lower() == ".tsv":
            delimiter = "\t"
        captions = []
        with open(caption_path, "r", encoding="utf-8", newline="") as file_obj:
            reader = csv.reader(file_obj, delimiter=delimiter)
            if self.caption_has_header:
                next(reader, None)
            for row in reader:
                if self.caption_filter_column is not None and self.caption_filter_value is not None:
                    row_value = row[self.caption_filter_column].strip().lower()
                    target_value = self.caption_filter_value.strip().lower()
                    if row_value != target_value:
                        continue
                caption = row[self.caption_column].strip()
                if caption:
                    captions.append(caption)
        return captions

    def read_text_captions(self, caption_path: Path) -> list[str]:
        with open(caption_path, "r", encoding="utf-8") as file_obj:
            return [line.strip() for line in file_obj if line.strip()]
