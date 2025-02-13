import argparse
from distance_functions import levenshtein_distance, jaccard_distance, hamming_distance
from utility import read_sample_from_file
from learn_dfa import learn_minimal_dfa, learn_distance_based_dfa


def main():
    parser = argparse.ArgumentParser(description="Learn a DFA from a sample.")
    parser.add_argument("--filepath", type=str, default="sample.txt", help="Path to the sample file.")
    parser.add_argument("--algorithm", type=str, choices=["1", "2", "3"], help="Algorithm to use.")
    parser.add_argument("--distance", type=str, choices=["lev", "jacc", "ham"], default="lev",
                        help="Distance function to use.")
    parser.add_argument("--outlier_weight", type=str, choices=["lev", "jacc", "ham"], default="1",
                        help="Wheights given to outliers in contrast to regular objects "
                             "(used in distance based approach).")
    parser.add_argument("--lower_bound", type=int, help="Lower bound for the number of accepted words.")
    parser.add_argument("--upper_bound", type=int, help="Upper bound for the number of accepted words.")
    parser.add_argument("--min_dfa_size", type=int, default=1, help="Minimum DFA size.")
    parser.add_argument("--visualize", action="store_true", help="Display and save visualization.")
    parser.add_argument("--save_path", type=str, default="./dfa.png", help="Path to save the visualization.")

    args = parser.parse_args()

    try:
        sample, alphabet = read_sample_from_file(args.filepath)
    except FileNotFoundError:
        print(f"File {args.filepath} not found.")
        return

    # Choose distance function based on the argument
    distance_funcs = {
        "lev": levenshtein_distance,
        "jacc": jaccard_distance,
        "ham": hamming_distance
    }
    distance_func = distance_funcs[args.distance]

    dfa = None
    if args.algorithm == "1":
        dfa = learn_minimal_dfa(sample, alphabet, args.lower_bound, args.upper_bound, args.min_dfa_size)
    elif args.algorithm == "2":
        dfa = learn_minimal_dfa(sample, alphabet, args.lower_bound, args.upper_bound, args.min_dfa_size)
    elif args.algorithm == "3":
        dfa = learn_distance_based_dfa(sample, alphabet, args.min_dfa_size, distance_func)

    if dfa:
        print("Minimal DFA found:")
        print(f"States: {dfa['states']}")
        print(f"Alphabet: {dfa['alphabet']}")
        print(f"Transitions: {dfa['transitions']}")
        print(f"Initial State: {dfa['initial_state']}")
        print(f"Final States: {dfa['final_states']}")
    else:
        print("No feasible DFA found within the given bounds.")

    def evaluate_sample_with_dfa(sample, dfa):
        accepted = []
        rejected = []

        transitions = dfa['transitions']
        initial_state = dfa['initial_state']
        final_states = dfa['final_states']

        for word in sample:
            current_state = initial_state
            accepted_word = True

            for symbol in word:
                if (current_state, symbol) in transitions:
                    current_state = transitions[(current_state, symbol)]
                else:
                    accepted_word = (current_state in final_states)
                    break

            if accepted_word and current_state in final_states:
                accepted.append(word)
            else:
                rejected.append(word)

        return accepted, rejected

    accepted, rejected = evaluate_sample_with_dfa(sample, dfa)

    print("Accepted words:", accepted)
    print("Rejected words:", rejected)


if __name__ == "__main__":
    main()