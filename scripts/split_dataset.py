import argparse
import multiprocessing
from pathlib import Path

import dgl
import joblib as J
import numpy as np
import pandas as pd


def extract_stats(file: str):
    file = Path(file)
    if not file.exists():
        raise ValueError(f"{file} doesn't exist")
    result = {}
    graphs, labels = dgl.data.utils.load_graphs(str(file))
    graph: dgl.DGLGraph = graphs[0]
    result['label'] = 'Benign' if 'Benig' in file.stem else 'Malware'
    result['file_name'] = str(file)
    result['num_nodes'] = graph.num_nodes()
    result['num_edges'] = graph.num_edges()
    return result


def save_list(dataframe, file_name):
    with open(file_name, 'a') as target:
        for file in dataframe['file_name']:
            target.writelines(f'{file.split(".")[0]}\n')


def get_dataset(df: pd.DataFrame, test_ratio: float, log_dir: Path):
    assert 0 <= test_ratio < 1, "Ratio must be within 0 and 1"
    q1 = df['num_nodes'].quantile(0.25)
    q3 = df['num_nodes'].quantile(0.75)
    iqr = q3 - q1
    print(f"Initial range {df['num_nodes'].min(), df['num_nodes'].max()}")
    print(f"IQR num_nodes = {iqr}")
    df = df.query(f'{q1 - iqr} <= num_nodes <= {q3 + iqr}')
    print(f"Final range {df['num_nodes'].min(), df['num_nodes'].max()}")
    bins = np.arange(0, df['num_nodes'].max(), 500)
    ben_hist, _ = np.histogram(df.query('label == "Benign"')['num_nodes'], bins=bins)
    mal_hist, _ = np.histogram(df.query('label != "Benign"')['num_nodes'], bins=bins)
    combined = np.concatenate([ben_hist[:, np.newaxis], mal_hist[:, np.newaxis]], axis=1)
    np.savetxt(
        log_dir / 'histogram.list',
        combined
    )
    final_sizes = [(x, x) for x in np.min(combined, axis=1)]
    final_train = []
    final_test = []
    for i, (ben_size, mal_size) in enumerate(final_sizes):
        low, high = bins[i], bins[i + 1]
        benign_samples = df.query(f'label == "Benign" and {low} <= num_nodes < {high}')
        malware_samples = df.query(f'label == "Malware" and {low} <= num_nodes < {high}')
        assert len(benign_samples) >= ben_size and len(malware_samples) >= mal_size, "Mismatch"
        benign_samples = benign_samples.sample(ben_size)
        malware_samples = malware_samples.sample(mal_size)
        if test_ratio > 0:
            benign_samples, benign_test_samples = np.split(benign_samples,
                                                           [round((1 - test_ratio) * len(benign_samples))])
            malware_samples, malware_test_samples = np.split(malware_samples,
                                                             [round((1 - test_ratio) * len(malware_samples))])
            final_test.append(benign_test_samples)
            final_test.append(malware_test_samples)
        final_train.append(benign_samples)
        final_train.append(malware_samples)
    final_train = pd.concat(final_train)
    if final_test:
        final_test = pd.concat(final_test)
    return final_train, final_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split the input dataset into train and test partitions (80%, 20%) based on bin equalization'
    )
    parser.add_argument(
        '-i', '--input-dirs',
        help="List of input paths",
        nargs='+',
        required=True
    )
    parser.add_argument(
        '-o', '--output-dir',
        help="The path to write the result lists to",
        required=True
    )
    parser.add_argument(
        '-s', '--strict',
        help="If set, program will terminate on error while in loop",
        action='store_true',
        default=False
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    input_stats = []
    for input_dir in args.input_dirs:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            if args.strict:
                raise FileNotFoundError(f"{input_dir} does not exist. Halting")
            else:
                print(f"{input_dir} does not exist. Skipping...")
                continue
        stats = J.Parallel(n_jobs=multiprocessing.cpu_count())(
            J.delayed(extract_stats)(x) for x in input_dir.glob("*.fcg")
        )
        input_stats.append(pd.DataFrame.from_records(stats))
    input_stats = pd.concat(input_stats)
    zero_nodes = input_stats.query('num_nodes == 0')
    if len(zero_nodes) > 0:
        print(f"Warning: {len(zero_nodes)} APKs with num_nodes = 0 found. Writing their names to zero_nodes.list")
        save_list(zero_nodes, 'zero_nodes.list')
    input_stats = input_stats.query('num_nodes != 0')
    train_list, test_list = get_dataset(input_stats, 0.2, output_dir)
    save_list(train_list, 'train.list')
    save_list(test_list, 'test.list')
