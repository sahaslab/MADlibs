import csv
import json
if __name__ == "__main__":
    TEST = True

    sdd_real_mir_filename = "../../shared/MADLibs_output/sdd/real/mir/mir_results.jsonl"
    alignment_csv_filename = "data/song_describer.csv"
    new_mir_filename = "../../shared/MADLibs_output/sdd/real/mir/expanded_mir_results.jsonl"

    mir_data = []
    with open(sdd_real_mir_filename, 'r') as file:
        mir_data = [json.loads(line) for line in file]

    print('original mir length:', len(mir_data))

    new_mir_data = []
    mir_filename_pointer = -1
    prev_alignment_track_id = None

    with open(alignment_csv_filename, mode='r') as file:
        csv_file = csv.reader(file)
        for line in csv_file:
            track_id = line[1]
            if track_id == 'track_id':
                # skip first line
                continue

            if track_id != prev_alignment_track_id:
                # we need to move to the next mir results line
                mir_filename_pointer += 1
            
            if TEST:
                # Flag set in the top of file
                # Make sure that the track ids line up
                audio_filename = mir_data[mir_filename_pointer]["metadata"]["audio_filename"]
                audio_track_id = audio_filename.split('/')[-1].split('.')[0]
                if track_id != audio_track_id:
                    print('error for track id', track_id)


            new_mir_data.append(mir_data[mir_filename_pointer])
            prev_alignment_track_id = track_id

    print('new mir length:', len(new_mir_data))

    with open(new_mir_filename, 'w') as file:
        for line in new_mir_data:
            file.write(json.dumps(line) + '\n')