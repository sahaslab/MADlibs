import csv
import nltk
from nltk import word_tokenize, pos_tag

if __name__ == "__main__":
    SDD_PATH = "src/scripts/song_describer.csv"

    nltk.download("punkt")
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download("averaged_perceptron_tagger")

    # POS MAP from ChatGPT
    POS_MAP = {
        "noun": {"NN", "NNS", "NNP", "NNPS"},
        "verb": {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"},
        "adj": {"JJ", "JJR", "JJS"},
        "adv": {"RB", "RBR", "RBS"},
        "relative_adv": {"WRB"},
        "article": {"DT"},
        "pronoun": {"PRP", "PRP$", "WP", "WP$"},
        "conjunction": {"CC", "IN"},
    }

    pos = {
        "noun": set(),
        "verb": set(),
        "adv": set(),
        "adj": set(),
        "relative_adv": set(),
        "article": set(),
        "pronoun": set(),
        "conjunction": set()
    }


    with open(SDD_PATH, mode='r') as file:
        csvFile = csv.reader(file)

        for line in csvFile:
            desc = line[2]
            
            tokens = word_tokenize(desc)
            tagged = pos_tag(tokens)

            for word, tag in tagged:
                word_lower = word.lower()

                for category, allowed_tags in POS_MAP.items():
                    if tag in allowed_tags:
                        pos[category].add(word_lower)
                        break

    with open('./parts_of_speech.txt', mode='w') as file:
        for part_of_speech, speech_list in pos.items():
            speech_list_line = ", ".join(speech_list)
            file.write(f"*{part_of_speech}*\n")
            file.write(f"{speech_list_line}\n")
