import argparse


# function to read out all the command line arguments 
def fit_arguments():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
    "--paths",
    metavar="PATH",
    type=str,
    nargs="*",
    default=[],
    help="Choose input file(s). Could either be a pickle file (factory) or (many) hdf5 files",
    )

    parser.add_argument("--plot", "-p", action="store_true", help="Plot results (default: false)")

    parser.add_argument(
        "--save",
        metavar="file name",
        type=str,
        nargs="?",
        default=False,
        const=True,
        help="Save results in a factory (pickled). Default is false, const will store result in input file, argument: name of new pickled file",
    )

    parser.add_argument(
        "--fit_results", action="store_true", help="Plot results (default: false)"
    )

    parser.add_argument(
        "--label", metavar="label", type=str, nargs="?", default="", help="Set the label"
    )

    parser.add_argument(
        "--parallel_jobs",
        "-pj",
        metavar="number of jobs",
        type=int,
        nargs="?",
        default=24,
        help="Number of ray jobs/workers (default: 24)",
    )

    # parser asks for atmosphere model to use in calculation
    parser.add_argument(
        "-m",
        "--atmModel",
        metavar="int",
        type=int,
        nargs="?",
        default=1,
        help="Set the model id for atmospheric model",
    )

    # parser asks for atmosphere model to use in calculation
    parser.add_argument(
        "-gd",
        "--gdasFile",
        metavar="PATH",
        type=str,
        nargs="*",
        default=None,
        help="Choose gdas atmosphere file.",
    )

    parser.add_argument(
        "-f",
        "--function",
        metavar="function",
        type=str,
        nargs="?",
        default=None,
        help="Set a function to run",
    )

    parser.add_argument(
        "-s",
        "--shower",
        metavar="shower_type",
        type=str,
        nargs="?",
        default="sim_shower",
        help="Set the shower type which is used by the functions (default: rd_shower)",
    )

    parser.add_argument("--verbose", action="store_true", help="(default: false)")

    # event selection
    parser.add_argument(
        "-id",
        "--event_id",
        nargs="*",
        # type=int,
        default=None,
        help="Select only certain events (default: None (all))",
    )

    parser.add_argument(
        "--thinning_cut", nargs="?", type=float, default=0.85, help="(default: 0.85)"
    )

    parser.add_argument("-real", "--realistic_input", action="store_true", help="(default: false)")

    parser.add_argument(
        "-rmif",
        "--remove_infill",
        action="store_true",
        default=None,
        help="Whether to remove infill antennas(default: False)"
    )

    parser.add_argument(
        "-onif",
        "--only_infill",
        action="store_true",
        default=None,
        help="Whether to only use in-fill antennas(default: False)"
    )

    args = parser.parse_args()
    return args