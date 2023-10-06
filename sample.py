from tap import Tap
import argparse


class SimpleArgumentParser(Tap):
    name: str  # Your name
    # # # language: str = "Python"  # Programming language
    # # # package: str = "Tap"  # Package name
    stars: int | None = None  # Number of stars
    counts: list[int]

    # # max_stars: int = 5  # Maximum stars
    # def configure(self):
    #     self.add_argument('name')
    #     self.add_argument("--stars")


def parse_parameters():
    return SimpleArgumentParser().parse_args(known_only=True)

    # print(f"My name is {args.name}  {args.stars}")


def parse_parameters_without_type():
    """_summary_
    引数をパースする
    Args:
        parser (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    parser = argparse.ArgumentParser()

    ## Data
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--stars", default=None, type=int)

    args, cant_parsed = parser.parse_known_args()
    print(args, cant_parsed)

    return args


if __name__ == "__main__":
    args = parse_parameters()
    # args = parse_parameters_without_type()
    print(args.name, args.stars)
    # print("ok")
