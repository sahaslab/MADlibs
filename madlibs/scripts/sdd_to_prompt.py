import csv

if __name__ == "__main__":
    output_file = "/dartfs/rc/lab/S/SinghN/shared/MADLibs_output/sdd/prompt.txt"
    sdd_path = "data/song_describer.csv"

    prompt_str = ""
    with open(sdd_path, mode="r") as file:
        csv_f = csv.reader(file)
        for line in csv_f:
            if line[2] == "caption":
                # ignore first line
                continue
            prompt_str += (line[2] + '\n')
        file.close()

    with open(output_file, mode="w") as file:
        file.write(prompt_str)
        file.close()