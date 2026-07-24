from stock_research.indicators import preprocess_folder
from stock_research.paths import load_paths


def main() -> None:
    paths = load_paths()
    outputs = preprocess_folder(paths.raw_prices, paths.processed)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
