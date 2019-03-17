from experiments.encoding_state import encodingState


def main():
    state_model = encodingState()
    state_model.train()


if __name__ == "__main__":
    main()