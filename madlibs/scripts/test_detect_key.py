from madlibs.mir.key_detector import KeyDetector
import torch
import statistics

if __name__ == "__main__":
    key_detector = KeyDetector(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    keys, positions = key_detector.detect_key('/dartfs/rc/lab/S/SinghN/noah/imPROMPTu/LMM-Gen-Distribution/tests/mir/ace_step_test.wav', 60)
    print(keys, positions)
    print(statistics.mode(keys))