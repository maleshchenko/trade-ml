from trade_model import LSTMModel, load_checkpoint, stream_live_signals


def main():
    model = LSTMModel()
    means, stds = load_checkpoint(model)
    stream_live_signals(model, means, stds)


if __name__ == "__main__":
    main()
