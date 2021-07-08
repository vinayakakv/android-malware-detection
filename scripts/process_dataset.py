import argparse
import json
import multiprocessing
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union, Optional

import dgl
import joblib as J
import networkx as nx
import torch
from androguard.core.analysis.analysis import MethodAnalysis
from androguard.core.api_specific_resources import load_permission_mappings
from androguard.misc import AnalyzeAPK
from pygtrie import StringTrie

ATTRIBUTES = ['external', 'entrypoint', 'native', 'public', 'static', 'codesize', 'api', 'user']
package_directory = os.path.dirname(os.path.abspath(__file__))

stats: Dict[str, int] = defaultdict(int)


def memoize(function):
    """
    Alternative to @lru_cache which could not be pickled in ray
    :param function: Function to be cached
    :return: Wrapped function
    """
    memo = {}

    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv

    return wrapper


class FeatureExtractors:
    NUM_PERMISSION_GROUPS = 20
    NUM_API_PACKAGES = 226
    NUM_OPCODE_MAPPINGS = 21

    @staticmethod
    def _get_opcode_mapping() -> Dict[str, int]:
        """
        Group opcodes and assign them an ID
        :return: Mapping from opcode group name to their ID
        """
        mapping = {x: i for i, x in enumerate(['nop', 'mov', 'return',
                                               'const', 'monitor', 'check-cast', 'instanceof', 'new',
                                               'fill', 'throw', 'goto/switch', 'cmp', 'if', 'unused',
                                               'arrayop', 'instanceop', 'staticop', 'invoke',
                                               'unaryop', 'binop', 'inline'])}
        mapping['invalid'] = -1
        return mapping

    @staticmethod
    @memoize
    def _get_instruction_type(op_value: int) -> str:
        """
        Get instruction group name from instruction
        :param op_value: Opcode value
        :return: String containing ID of :instr:
        """
        if 0x00 == op_value:
            return 'nop'
        elif 0x01 <= op_value <= 0x0D:
            return 'mov'
        elif 0x0E <= op_value <= 0x11:
            return 'return'
        elif 0x12 <= op_value <= 0x1C:
            return 'const'
        elif 0x1D <= op_value <= 0x1E:
            return 'monitor'
        elif 0x1F == op_value:
            return 'check-cast'
        elif 0x20 == op_value:
            return 'instanceof'
        elif 0x22 <= op_value <= 0x23:
            return 'new'
        elif 0x24 <= op_value <= 0x26:
            return 'fill'
        elif 0x27 == op_value:
            return 'throw'
        elif 0x28 <= op_value <= 0x2C:
            return 'goto/switch'
        elif 0x2D <= op_value <= 0x31:
            return 'cmp'
        elif 0x32 <= op_value <= 0x3D:
            return 'if'
        elif (0x3E <= op_value <= 0x43) or (op_value == 0x73) or (0x79 <= op_value <= 0x7A) or (
                0xE3 <= op_value <= 0xED):
            return 'unused'
        elif (0x44 <= op_value <= 0x51) or (op_value == 0x21):
            return 'arrayop'
        elif (0x52 <= op_value <= 0x5F) or (0xF2 <= op_value <= 0xF7):
            return 'instanceop'
        elif 0x60 <= op_value <= 0x6D:
            return 'staticop'
        elif (0x6E <= op_value <= 0x72) or (0x74 <= op_value <= 0x78) or (0xF0 == op_value) or (
                0xF8 <= op_value <= 0xFB):
            return 'invoke'
        elif 0x7B <= op_value <= 0x8F:
            return 'unaryop'
        elif 0x90 <= op_value <= 0xE2:
            return 'binop'
        elif 0xEE == op_value:
            return 'inline'
        else:
            return 'invalid'

    @staticmethod
    def _mapping_to_bitstring(mapping: List[int], max_len) -> torch.Tensor:
        """
        Convert opcode mappings to bitstring
        :param max_len:
        :param mapping: List of IDs of opcode groups (present in an method)
        :return: Binary tensor of length `len(opcode_mapping)` with value 1 at positions specified by :poram mapping:
        """
        size = torch.Size([1, max_len])
        if len(mapping) > 0:
            indices = torch.LongTensor([[0, x] for x in mapping]).t()
            values = torch.LongTensor([1] * len(mapping))
            tensor = torch.sparse.LongTensor(indices, values, size)
        else:
            tensor = torch.sparse.LongTensor(size)
        # Sparse tensor is normal tensor on CPU!
        return tensor.to_dense().squeeze()

    @staticmethod
    def _get_api_trie() -> StringTrie:
        apis = open(Path(package_directory).parent / "metadata" / "api.list").readlines()
        api_list = {x.strip(): i for i, x in enumerate(apis)}
        api_trie = StringTrie(separator='.')
        for k, v in api_list.items():
            api_trie[k] = v
        return api_trie

    @staticmethod
    @memoize
    def get_api_features(api: MethodAnalysis) -> Optional[torch.Tensor]:
        if not api.is_external():
            return None
        api_trie = FeatureExtractors._get_api_trie()
        name = str(api.class_name)[1:-1].replace('/', '.')
        _, index = api_trie.longest_prefix(name)
        if index is None:
            indices = []
        else:
            indices = [index]
        feature_vector = FeatureExtractors._mapping_to_bitstring(indices, FeatureExtractors.NUM_API_PACKAGES)
        return feature_vector

    @staticmethod
    @memoize
    def get_user_features(user: MethodAnalysis) -> Optional[torch.Tensor]:
        if user.is_external():
            return None
        opcode_mapping = FeatureExtractors._get_opcode_mapping()
        opcode_groups = set()
        for instr in user.get_method().get_instructions():
            instruction_type = FeatureExtractors._get_instruction_type(instr.get_op_value())
            instruction_id = opcode_mapping[instruction_type]
            if instruction_id >= 0:
                opcode_groups.add(instruction_id)
        # 1 subtraction for 'invalid' opcode group
        feature_vector = FeatureExtractors._mapping_to_bitstring(list(opcode_groups), len(opcode_mapping) - 1)
        return torch.LongTensor(feature_vector)


def process(source_file: Path, dest_dir: Path):
    try:
        file_name = source_file.stem
        _, _, dx = AnalyzeAPK(source_file)
        cg = dx.get_call_graph()
        mappings = {}
        for node in cg.nodes():
            features = {
                "api": torch.zeros(FeatureExtractors.NUM_API_PACKAGES),
                "user": torch.zeros(FeatureExtractors.NUM_OPCODE_MAPPINGS)
            }
            if node.is_external():
                features["api"] = FeatureExtractors.get_api_features(node)
            else:
                features["user"] = FeatureExtractors.get_user_features(node)
            mappings[node] = features
        nx.set_node_attributes(cg, mappings)
        cg = nx.convert_node_labels_to_integers(cg)
        dg = dgl.from_networkx(cg, node_attrs=ATTRIBUTES)
        dest_dir = dest_dir / f'{file_name}.fcg'
        dgl.data.utils.save_graphs(str(dest_dir), [dg])
        print(f"Processed {source_file}")
    except:
        print(f"Error while processing {source_file}")
        traceback.print_exception(*sys.exc_info())
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess APK Dataset into Graphs')
    parser.add_argument(
        '-s', '--source-dir',
        help='The directory containing apks',
        required=True
    )
    parser.add_argument(
        '-d', '--dest-dir',
        help='The directory to store processed graphs',
        required=True
    )
    parser.add_argument(
        '--override',
        help='Override existing processed files',
        action='store_true'
    )
    parser.add_argument(
        '--dry',
        help='Run without actual processing',
        action='store_true'
    )
    parser.add_argument(
        '--n-jobs',
        default=multiprocessing.cpu_count(),
        help='Number of jobs to be used for processing'
    )
    parser.add_argument(
        '--limit',
        help='Run for n apks',
        default=-1
    )
    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f'{source_dir} not found')
    dest_dir = Path(args.dest_dir)
    if not dest_dir.exists():
        raise FileNotFoundError(f'{dest_dir} not found')
    n_jobs = args.n_jobs
    if n_jobs < 2:
        print(f"n_jobs={n_jobs} is too less. Switching to number of CPUs in this machine instead")
        n_jobs = multiprocessing.cpu_count()
    files = [x for x in source_dir.iterdir() if x.is_file()]
    source_files = set([x.stem for x in files])
    dest_files = set([x.name for x in dest_dir.iterdir() if x.is_file()])
    unprocessed = [source_dir / f'{x}.apk' for x in source_files - dest_files]
    print(f"Only {len(unprocessed)} out of {len(source_files)} remain to be processed")
    if args.override:
        print(f"--override specified. Ignoring {len(source_files) - len(unprocessed)} processed files")
        unprocessed = [source_dir / f'{x}.apk' for x in source_files]
    print(f"Starting dataset processing with {n_jobs} Jobs")
    limit = int(args.limit)
    if limit != -1:
        print(f"Limiting dataset processing to {limit} apks.")
        unprocessed = unprocessed[:limit]
    if not args.dry:
        J.Parallel(n_jobs=n_jobs)(J.delayed(process)(x, dest_dir) for x in unprocessed)
    print("DONE")
