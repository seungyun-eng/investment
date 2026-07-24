from stock_research.macro_data import update_macro_data
from stock_research.paths import load_paths


def main() -> None:
    paths = load_paths()
    outputs = update_macro_data(paths.macro)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
