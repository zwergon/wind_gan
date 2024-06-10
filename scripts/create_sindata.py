import argparse
import numpy as np
import argparse


def sine_data_generation(no, seq_len):
    """Sine data generation.

    Args:
        - no: the number of samples
        - seq_len: sequence length of the time-series
        - dim: feature dimensions

    Returns:
        - data: generated data
    """
    # Initialize the output
    data = np.zeros(shape=(no, seq_len + 2), dtype=np.float32)

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature

        # Randomly drawn frequency and phase
        freq = np.random.uniform(0, 0.1)
        data[i, -1] = freq
        phase = np.random.uniform(0, 0.1)
        data[i, -2] = phase
        
        # Generate sine signal based on the drawn frequency and phase
        for j in range(seq_len):
            data[i, j] = np.sin(freq * j + phase)

    # normalize
    data[:, :-2] = (data[:, :-2] + 1) * 0.5

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse two optional float arguments: number of samples and sequence length."
    )

    parser.add_argument('outname', help='filename to store csv file')

    # Ajout de l'argument optionnel num_samples
    parser.add_argument("-n",
        "--num_samples", type=int, default=1000, help="Number of samples."
    )

    # Ajout de l'argument optionnel sequence_length
    parser.add_argument(
        "-s", "--sequence_length",
        type=int,
        default=1200,
        help="Length of the sequence.",
    )

    args = parser.parse_args()

    print(f"Number of samples: {args.num_samples}")
    print(f"Length of the sequence: {args.sequence_length}")

    data = sine_data_generation(args.num_samples, args.sequence_length)
    np.savetxt(args.outname, data, delimiter=',')