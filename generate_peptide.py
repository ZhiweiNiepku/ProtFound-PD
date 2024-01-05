import os
import math
import numpy as np
from mindspore.parallel.nn import TransformerOpParallelConfig
import mindspore.common.dtype as mstype
import mindspore.communication.management as D
from mindspore import context, Tensor
from mindspore import export
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.train.model import Model
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.train.serialization import load_distributed_checkpoint
from src.pangu_alpha import PanguAlphaModel, EvalNet
from src.pangu_alpha_config import PanguAlphaConfig, set_parse
from src.utils import get_args
from src.utils import download_data
import math
import random

os.environ.pop('CREDENTIAL_PROFILES_FILE', None)
os.environ.pop('AWS_SHARED_CREDENTIALS_FILE', None)
base_path = os.path.split(os.path.abspath(__file__))[0]
output_path = base_path + "/output/"
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
project_root = os.path.abspath(
    os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


class AMPDataset:
    def __init__(self, dataset_path, num_samples):
        f = open(dataset_path, 'r')
        self.data = f.readlines()
        self.num_samples = num_samples
        self.max_len = 8
        self.map_dict = {'L': 5, 'S': 6, 'A': 7, 'G': 8, 'E': 9, 'V': 10, 'T': 11, 'R': 12, 'D': 13, 'I': 14,
                         'P': 15, 'K': 16, 'N': 17, 'F': 18, 'Q': 19, 'Y': 20, 'H': 21, 'M': 22, 'C': 23, 'W': 24,
                         'X': 1, 'B': 2, 'Z': 3, 'U': 4, 'O': 26, 'SOT': 25, 'EOT': 0, 'PAD': 0}

    def __getitem__(self, index):
        seq = self.data[index].strip('\n')
        if len(seq) >= self.max_len:
            seq = seq[:self.max_len - 1]
        labels = [self.map_dict[i] for i in list(seq)] + [self.map_dict['EOT']]
        inputs = [self.map_dict['SOT']] + [self.map_dict[i] for i in list(seq)]
        inputs = np.array(inputs, np.int32)
        labels = np.array(labels, np.int32)
        positions = np.arange(self.max_len).astype(np.int32)
        item = (inputs, labels, positions)
        return item

    def __len__(self):
        return self.num_samples


class Sampler:
    def __init__(self, num_data, local_rank, world_size):
        self.__num_data = num_data
        self.__local_rank = local_rank
        self.__world_size = world_size
        self.samples_per_rank = int(math.ceil(self.__num_data / float(self.__world_size)))
        self.total_num_samples = self.samples_per_rank * self.__world_size

    def __iter__(self):
        indices = range(self.__local_rank, self.total_num_samples, self.__world_size)
        return iter(indices)

    def __len__(self):
        return self.samples_per_rank

    def __del__(self):
        for f in self.f_list:
            f.close()


def load_model(args_opt):
    r"""
     The main function for load model
    """
    # Set execution mode
    context.set_context(save_graphs=False,
                        mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target)
    context.set_context(variable_memory_max_size="30GB")
    # Set parallel context
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        print("rank_id is {}, device_num is {}".format(rank, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            full_batch=True,
            loss_repeated_mean=True,
            enable_parallel_optimizer=False,
            strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path)
        # pipeline_stages=args_opt.stage_num)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()

    else:
        rank = 0
        device_num = 1
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path)

    use_past = (args_opt.use_past == "true")
    print('local_rank:{}, start to run...'.format(rank), flush=True)
    if args_opt.export:
        use_past = True
    # Set model property
    model_parallel_num = args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    per_batch_size = args_opt.per_batch_size
    batch_size = 1  # per_batch_size * data_parallel_num
    # Now only support single batch_size for predict
    if args_opt.run_type == "predict":
        batch_size = 1
    parallel_config = TransformerOpParallelConfig(data_parallel=1,
                                                  model_parallel=1,
                                                  pipeline_stage=args_opt.stage_num,
                                                  micro_batch_num=args_opt.micro_size,
                                                  optimizer_shard=False,
                                                  vocab_emb_dp=bool(args_opt.word_emb_dp),
                                                  recompute=True)
    config = PanguAlphaConfig(batch_size=1,
                              seq_length=128,
                              vocab_size=args_opt.vocab_size,
                              hidden_size=args_opt.embedding_size,
                              ffn_hidden_size=args_opt.embedding_size * 4,
                              num_layers=args_opt.num_layers,
                              num_heads=args_opt.num_heads,
                              load_ckpt_path=args_opt.load_ckpt_path,
                              param_init_type=mstype.float32 if args_opt.param_init_type == 'fp32' else mstype.float16,
                              post_layernorm_residual=False,
                              dropout_rate=0.0,
                              eod_token=0,
                              use_past=False,
                              hidden_act='gelu',
                              eod_reset=False,
                              enable_offload=False,
                              parallel_config=parallel_config)

    print("===config is: ", config, flush=True)
    print("=====args_opt is: ", args_opt, flush=True)
    pangu_alpha = PanguAlphaModel(config=config)
    param_dict = load_checkpoint(args_opt.load_ckpt_path)
    load_param_into_net(pangu_alpha, param_dict)
    print("ckpt load succeed!")
    eval_net = EvalNet(pangu_alpha)
    eval_net.set_train(False)
    model_predict = Model(eval_net)
    # Compile network and obtain tensor layout for loading ckpt
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    current_index = Tensor(np.array([0]), mstype.int32)

    if args_opt.distribute == "false":
        predict_layout = None
    elif config.use_past:
        batch_valid_length = Tensor(np.array([0]), mstype.int32)
        init_true = Tensor([True], mstype.bool_)
        inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index, init_true, batch_valid_length)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
        _ = model_predict.infer_predict_layout(inputs_np_1, current_index, init_true, batch_valid_length)
    else:
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index)
    return model_predict, config


def export_mindir(model_predict, config):
    """Export mindir model"""
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    current_index = Tensor(np.array([0]), mstype.int32)

    batch_valid_length = Tensor(np.array([0]), mstype.int32)
    init_true = Tensor([True], mstype.bool_)
    inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)

    model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
    export(model_predict.predict_network, inputs_np, current_index,
           init_true, batch_valid_length, file_name='pangu_alpha_1024', file_format='MINDIR')
    model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
    export(model_predict.predict_network, inputs_np_1, current_index,
           init_true, batch_valid_length, file_name='pangu_alpha_1', file_format='MINDIR')
    print("Export finished and now exit.")


def run_predict(model_predict, config, args_opt):
    """run predict"""
    a_dict = {'L': 5, 'S': 6, 'A': 7, 'G': 8, 'E': 9, 'V': 10, 'T': 11, 'R': 12, 'D': 13, 'I': 14,
              'P': 15, 'K': 16, 'N': 17, 'F': 18, 'Q': 19, 'Y': 20, 'H': 21, 'M': 22, 'C': 23, 'W': 24,
              'X': 1, 'B': 2, 'Z': 3, 'U': 4, 'O': 26, 'SOT': 25, 'SHT': 27, 'MED': 28, 'LON': 29, 'EOT': 0, 'PAD': 0}
    dic = {value: key for key, value in a_dict.items()}
    from src.generate import generate
    f = open(args_opt.head_txt, 'r')
    count = 0
    count_1 = 0
    head_list = []
    total_list = []
    for line in f:
        seq = line.strip('\n')
        head_list.append(seq)
    total_list.extend(head_list)

    for seq in total_list:
        org_seq = seq.strip("\n")
        seq = seq.strip("\n")
        mutation_rate = random.random()
        change_count = math.floor(len(seq) * mutation_rate)
        change_index_list = []

        while len(change_index_list) < change_count:
            index = random.randint(0, len(seq) - 1)
            if index not in change_index_list and index != 0:
                change_index_list.append(index)
        change_index_list = sorted(change_index_list)
        for index in change_index_list:
            seq = seq[:index]
            inputs = [a_dict['SOT']] + [a_dict[i] for i in list(seq)]
            sentence = np.array(inputs).reshape(1, -1)
            cur_ss = a_dict[org_seq[index]]
            output_ids = generate(model_predict, sentence, args_opt, cur_ss, org_seq, index)
            output_samples = output_ids.tolist()
            output_str = ''
            for i in output_samples[1:]:
                if i not in [0, 25, 27, 28, 29, 30, 31]:
                    output_str += dic[i]
            output = output_str
            # print(output)
            if index < len(org_seq) - 1:
                seq = output + org_seq[index + 1:]
            else:
                seq = output
        count += 1
        count_1 += 1
        res = org_seq + "," + seq
        print('count:', count, 'Output is:', res, flush=True)


def main(opt):
    """Main process for predict or export model"""
    # opt = get_args(True)
    # set_parse(opt)
    model_predict, config = load_model(opt)
    #     if opt.export:
    #         export_mindir(model_predict, config)
    #     else:
    run_predict(model_predict, config, opt)


def file_name_walk(file_dir):
    print("file dir walk begin:{}".format(file_dir))
    for root, dirs, files in os.walk(file_dir):
        print("root", root)  
        print("dirs", dirs)  
        print("files", files)  
    print("file dir walk end:{}".format(file_dir))


if __name__ == "__main__":
    # main()
    opt = get_args(True)
    set_parse(opt)
    env_dist = os.environ
    rankid = int(os.environ['RANK_ID'])
    main(opt)
