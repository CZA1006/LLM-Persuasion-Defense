from src.utils import set_seed
from src.datasets import load_counterfact_subset
from src.strategies import ORDER_DEFAULT
from src.evaluate import evaluate_one

def main():
    set_seed(1337)
    # load 1 sample (toy or from JSONL if present)
    data = load_counterfact_subset(n=1)
    ex = data[0]

    res_no_def = evaluate_one(ex, ORDER_DEFAULT, defense=None)
    res_def    = evaluate_one(ex, ORDER_DEFAULT, defense="override")

    print("No-Defense:", res_no_def)
    print("Override  :", res_def)

if __name__ == "__main__":
    main()
