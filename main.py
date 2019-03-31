from algorithms.encoding_state import encodingState
from datasets_processing.SDF import SDF


def main():
    state_model = encodingState()
    state_model.train()
    # dataset = SDF()


if __name__ == "__main__":
    main()