import random

from src.prompt.IPrompt import IPrompt


class MadlibsPrompt(IPrompt):
    def __init__(self, part_of_speech_filename):
        # Parse part of speech file
        self.words = {}
        # Expect part of speech file to be in the format *POS* and then a list of those words in a comma separated value line
        with open(part_of_speech_filename, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        current_pos = None
        for line in lines:
            if line.startswith("*") and line.endswith("*"):
                current_pos = line.strip("*")
                self.words[current_pos] = []
            else:
                if current_pos is not None:
                    words = [w.strip() for w in line.split(",")]
                    self.words[current_pos].extend(words)

        # Generated templates with ChatGPT
        self.templates = [
            "{article} {adj} {noun} {verb}s.",
            "{article} {noun} {adv} {verb}s.",
            "{pronoun} {verb}s {adv}.",
            "{article} {noun} and {article} {noun} {verb}.",
            "{article} {noun} {verb}s when {pronoun} {verb}s.",
            "{article} {adj} {noun} {adv} {verb}s.",
            "{pronoun} {adv} {verb}s {article} {noun}.",
            "{article} {noun} {verb}s {conjunction} {pronoun} {verb}s.",
            "{article} {noun} {verb}s {relative_adv} {article} {noun} {verb}s.",
            "{pronoun} {verb}s {article} {adj} {noun}.",
            "{article} {noun} {adv} {verb}s {conjunction} {article} {noun} {verb}s.",
            "{article} {adj} {noun} {adv} {verb}s.",
            "{article} {noun} {verb}s {article} {adj} {noun}.",
            "{pronoun} {verb}s {article} {noun} {adv}.",
            "{article} {noun} {adv} {verb}s {relative_adv} {pronoun} {verb}s.",
            "{article} {adj} {noun} and {pronoun} {adv} {verb}s.",
            "{article} {noun} {verb}s {article} {noun}.",
            "{pronoun} {adv} {verb}s {article} {noun}.",
            "{article} {noun} {verb}s {conjunction} {pronoun} {verb}s.",
            "{article} {adj} {noun} {verb}s {relative_adv} {article} {adj} {noun} {verb}s.",
        ]

        self.prompt_starts = [
            "Please give me a song where ",
            "Song that ",
            "Write me a song where ",
        ]

    def get_prompts(self, n_prompts, seed=None):
        random.seed(seed)

        prompts = []

        for _ in range(n_prompts):
            prompt = random.choice(self.prompt_starts)

            template = random.choice(self.templates)

            prompt += template.format(
                article=random.choice(self.words["article"]),
                noun=random.choice(self.words["noun"]),
                verb=random.choice(self.words["verb"]),
                adv=random.choice(self.words["adv"]),
                adj=random.choice(self.words["adj"]),
                relative_adv=random.choice(self.words["relative_adv"]),
                pronoun=random.choice(self.words["pronoun"]),
                conjunction=random.choice(self.words["conjunction"]),
            )

            prompts.append(prompt)

        return prompts
