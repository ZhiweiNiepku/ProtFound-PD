import os
import math
import time
import numpy as np
import moxing as mox
import logging
from mindspore import context
from mindspore.train.model import Model
import mindspore.communication.management as D
from mindspore.context import ParallelMode
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore.train.callback import TimeMonitor, Callback, CheckpointConfig, ModelCheckpoint
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
import mindspore.common.dtype as mstype
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.nn.wrap.cell_wrapper import PipelineCell, _VirtualDatasetCell
from mindspore.parallel.nn import TransformerOpParallelConfig
from src.adam import AdamWeightDecayOp
from src.dataset import create_dataset
from src.pangu_alpha import PanGUAlphaWithLoss, PanguAlphaModel, CrossEntropyLoss
from src.pangu_alpha_wrapcell import PanguAlphaTrainOneStepWithLossScaleCell, PanguAlphaTrainPipelineWithLossScaleCell
from src.pangu_alpha_config import set_parse, PanguAlphaConfig
from src.utils import LearningRate, get_args, FP32StateAdamWeightDecay
from src.utils import download_data
from src.callbacks import EvalCallBack, LossCallBack
from src.metrics import PPLMetric
from mindspore.profiler import Profiler
from mindspore.train.serialization import load_checkpoint, load_param_into_net


base_path = os.path.split(os.path.abspath(__file__))[0]
output_path = base_path + "/output/"
profiling_path = base_path + '/profiling/'
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
if not os.path.exists(profiling_path):
    os.makedirs(profiling_path, exist_ok=True)
project_root = os.path.abspath(
    os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


class CkptUploadCallBack(Callback):
    """
    Upload checkpoint callback
    """
    def __init__(self, bucket_dir, max_ckpt=5, retry=3, retry_time=10, interval_num=128, interval_time=90, ckpt_save_steps=20000, local_file_path=None):
        self.bucket_dir = bucket_dir
        self.max_ckpt = max_ckpt
        self.retry = retry
        self.retry_time = retry_time
        self.interval_num = interval_num
        self.interval_time = interval_time
        self.local_file_path = local_file_path
        self.ckpt_save_steps = ckpt_save_steps

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_step_num = cb_params.cur_step_num
        if cur_step_num % self.ckpt_save_steps == 0:
            rank_id = os.getenv('RANK_ID')
            save_ckpt_sleep_time = 10
            time.sleep(save_ckpt_sleep_time)
            logging.info(f"rank_{rank_id} waits {save_ckpt_sleep_time}s for saving before uploading.")
            obs_dir = self.bucket_dir
            _, file_name = os.path.split(self.local_file_path)
            obs_file_path = os.path.join(self.bucket_dir, file_name)
            except_log_dir = os.path.join(self.bucket_dir, "upload_log")
            if not mox.file.exists(except_log_dir):
                mox.file.mk_dir(except_log_dir)

            # sleep due to restriction of obs
            sleep_time = int(rank_id) // self.interval_num * self.interval_time
            if sleep_time > 0:
                logging.info(f"rank_{rank_id} waits {sleep_time}s before uploading.")
                time.sleep(sleep_time)

            logging.info(f"rank_{rank_id}: start uploading {self.local_file_path} to {obs_file_path}.")
            if not mox.file.exists(obs_dir):
                mox.file.mk_dir(obs_dir)

            start = time.time()
            if os.path.exists(self.local_file_path):
                mox.file.copy_parallel(self.local_file_path, obs_file_path)
            else:
                print('File {} doest not exist.'.format(self.local_file_path))
            end = time.time()
            logging.info(f"rank_{rank_id}: uploading {self.local_file_path} to {obs_file_path} cost {end - start}s.")


def set_weight_decay(params):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-1
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    return group_params


def run_train(args_opt):
    r"""
    The main training process.
    """
    # Set execution mode
    args_opt.sink_size = 2
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(variable_memory_max_size="30GB")
    # Set parallel context
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        args_opt.optimizer_shard = 0
        print("rank_id is {}, device_num is {}".format(rank, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=False,
            full_batch=False, strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path,
            enable_parallel_optimizer=False) # , optimizer_weight_shard_size=128
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()
    else:
        rank = 0
        device_num = 1
    context.set_context(save_graphs=False, save_graphs_path=output_path + "./graphs_of_device_id_" + str(rank))
    # copy data from the cloud to the /cache/Data
    cache_url = '/cache/Data/'
    eval_cache_url = '/cache/EvalData/'
    if args_opt.offline:
        cache_url = args_opt.data_url
        eval_cache_url = args_opt.eval_data_url
    else:
        start_down_time = time.time()
        download_data(src_data_url=args_opt.data_url, tgt_data_path=cache_url, rank=rank)
        down_complete_time = time.time()
        print('Download Data Time: ', down_complete_time - start_down_time)
        # download_data(src_data_url=args_opt.eval_data_url, tgt_data_path=eval_cache_url, rank=rank)
    # Set model property
    model_parallel_num = args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    batch_size = args_opt.per_batch_size * data_parallel_num
    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num,
                                                  model_parallel=model_parallel_num,
                                                  pipeline_stage=args_opt.stage_num,
                                                  micro_batch_num=args_opt.micro_size,
                                                  optimizer_shard=False,
                                                  vocab_emb_dp=bool(args_opt.word_emb_dp),
                                                  recompute=True)
    config = PanguAlphaConfig(batch_size=batch_size, num_heads=args_opt.num_heads,
                              hidden_size=args_opt.embedding_size, seq_length=args_opt.seq_length,
                              vocab_size=args_opt.vocab_size, num_layers=args_opt.num_layers,
                              ffn_hidden_size=args_opt.embedding_size * 4,
                              eod_token=bool(args_opt.eod_reset),
                              load_ckpt_path=args_opt.load_ckpt_path,
                              param_init_type=mstype.float32 if args_opt.param_init_type == 'fp32' else mstype.float16,
                              enable_offload=bool(args_opt.opt_offload),
                              parallel_config=parallel_config)
    print("===config is: ", config, flush=True)
    # Define network
    pangu_alpha = PanguAlphaModel(config=config)
    loss = CrossEntropyLoss(parallel_config = config.parallel_config.dp_mp_config)
    pangu_alpha_with_loss_net = PanGUAlphaWithLoss(config, pangu_alpha, loss)
    pangu_alpha_with_loss = _VirtualDatasetCell(pangu_alpha_with_loss_net)

    print("=====args_opt is: ", args_opt, flush=True)
    
    if args_opt.finetune:
        # load ckpt
        cache_ckpt_url = "/cache/init_ckpt/model.ckpt"
        mox.file.copy_parallel(src_data_url=args_opt.load_ckpt_path, tgt_data_path=cache_ckpt_url)
        if device_num > 1 and rank == 0:
            params_dict = load_checkpoint(cache_ckpt_url)
            load_param_into_net(pangu_alpha_with_loss, params_dict)
    # Warm-up and cosine decay learning rate
    lr = LearningRate(learning_rate=args_opt.start_lr,
                      end_learning_rate=args_opt.end_lr,
                      warmup_steps=args_opt.warmup_step,
                      decay_steps=200000)

    params = pangu_alpha_with_loss.trainable_params()
    group_params = set_weight_decay(params)
    if args_opt.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    elif args_opt.opt_offload:
        optimizer = AdamWeightDecayOp(group_params, learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.95,
                                      param_init_type=config.param_init_type)
    else:
        optimizer = FP32StateAdamWeightDecay(group_params, learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.95)
    # Initial scaling sens
    loss_scale_value = math.pow(2, 32)
    epoch_num = args_opt.epoch_size
    # Dataset loading mindrecord files
    down_to_create = time.time()
    print('Download Data to Create DS Time: ', down_to_create - down_complete_time)
    dataset = create_dataset(config.batch_size, data_path=cache_url, data_start_index=0, eod_reset=config.eod_reset,
                        full_batch=False, eod_id=args_opt.eod_id, device_num=device_num,
                        rank=rank, column_name=args_opt.data_column_name, epoch=epoch_num)
    create_complete = time.time()
    print('Create Dataset Time: ', create_complete - down_to_create)

    actual_epoch_num = int(epoch_num * dataset.get_dataset_size() / args_opt.sink_size)
    callback = [TimeMonitor(args_opt.sink_size), LossCallBack(args_opt.sink_size, rank, 0, 0)]
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value, scale_factor=2, scale_window=1000)
    pangu_alpha_with_grads = PanguAlphaTrainOneStepWithLossScaleCell(
        pangu_alpha_with_loss, optimizer=optimizer, scale_update_cell=update_cell, enable_global_norm=True,
        config=config)

    if device_num == 1 or (device_num > 1 and rank == 0):
        agent_save_ckpt_path = output_path+"ckpt/"
        if not os.path.exists(agent_save_ckpt_path):
            os.makedirs(agent_save_ckpt_path, exist_ok=True)
        ckpt_cb = ModelCheckpoint(prefix='finetune_fc', directory=agent_save_ckpt_path,
                                  config=CheckpointConfig(save_checkpoint_steps=100,
                                                          keep_checkpoint_max=2))
        callback.append(ckpt_cb)
    if args_opt.train_and_eval_mode:
        ds_eval = create_dataset(config.batch_size, data_path=eval_cache_url,
                                 data_start_index=0, eod_reset=config.eod_reset, full_batch=bool(args_opt.full_batch),
                                 eod_id=args_opt.eod_id, device_num=device_num, rank=rank,
                                 column_name=args_opt.data_column_name, epoch=epoch_num,
                                 num_samples=args_opt.eval_steps * config.batch_size)
        ppl_metric = PPLMetric(config.seq_length)
        model = Model(pangu_alpha_with_grads, eval_network=pangu_alpha_with_loss, metrics={"ppl": ppl_metric})
        callback.append(EvalCallBack(model, ds_eval, ppl_metric))
    else:
        model = Model(pangu_alpha_with_grads)
    if args_opt.incremental_training:
        from mindspore.train.serialization import load_distributed_checkpoint
        strategy = model.infer_train_layout(train_dataset=ds, sink_size=args_opt.sink_size)
        print("======start load_distributed checkpoint", flush=True)
        # For 2.6B and 13B models, the number of ckpt files is 512.
        ckpt_file_list = [os.path.join(args_opt.load_ckpt_path, f"filerted_{ckpt_rank}.ckpt") for ckpt_rank in
                          range(0, 512)]
        print(f"Loading from path {ckpt_file_list[0]}", flush=True)
        load_distributed_checkpoint(model.train_network, ckpt_file_list, strategy)
    print("Dataset size: {}, actual_epoch_num: {}".format(dataset.get_dataset_size(), actual_epoch_num), flush=True)
    print('Start tload_ckpt_pathrain time', time.time)
    ds.config.set_sending_batches(4)
    model.train(6000, dataset, callbacks=callback, sink_size=args_opt.sink_size, dataset_sink_mode=True)
    # profiler.analyse()



def file_name_walk(file_dir):
    print("file dir walk begin:{}".format(file_dir))
    for root, dirs, files in os.walk(file_dir):
        print("root", root)  # now path
        print("dirs", dirs)  # sub dir list
        print("files", files)  # file list
    print("file dir walk end:{}".format(file_dir))

if __name__ == "__main__":
    os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
    os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
    os.environ["HCCL_CONNECT_TIMEOUT"] = "4800"

    env_dist = os.environ
    rankid = int(os.environ['RANK_ID'])

    cmd = "echo \"hello work work {}\" >> {}/test.txt".format(rankid, output_path)
    os.system(cmd)

    opt = get_args()
    set_parse(opt)
    print("lcm debug args_opt.num_layers:{}".format(opt.num_layers))
    print("train run  >>>>>>>>>>>>>>>>> begin rankid:{}".format(rankid))
    if opt.per_batch_size == 0:
        raise ValueError("The per_batch_size has not been configured.")
    try:
        if opt.stage_num > 1:
            run_train_pipeline(opt)
        else:
            run_train(opt)
    except Exception as e:
        print("run exception e:{}".format(e))
        print("lcm debug train run done copy done exception train_url:{} outpath:{}".format(opt.train_url, output_path))
        import moxing as mox
        import shutil
        if rankid % 8 == 0:
            file_name_walk(output_path)
            mox.file.copy_parallel(src_url=output_path, dst_url=opt.train_url)
            mox.file.copy_parallel(src_url=profiling_path, dst_url=opt.train_url)
            shutil.copy("/tmp/log/train.log", "/tmp/log/train.log"+str(rankid))
            mox.file.copy_parallel(src_url="/tmp/log/", dst_url=opt.train_url)
        print("train run  >>>>>>>>>>>>>>>>> rank-{} exception end".format(rankid))
    finally:
        print("lcm debug train run done copy done train_url:{} outpath:{}".format(opt.train_url, output_path))
        import moxing as mox
        import shutil
        if rankid % 8 == 0:
            file_name_walk(output_path)
            mox.file.copy_parallel(src_url=output_path, dst_url=opt.train_url)
            mox.file.copy_parallel(src_url=profiling_path, dst_url=opt.train_url)
            shutil.copy("/tmp/log/train.log", "/tmp/log/train.log"+str(rankid))
            mox.file.copy_parallel(src_url="/tmp/log/", dst_url=opt.train_url)
        print("lcm debug train run  >>>>>>>>>>>>>>>>> rank-{} end".format(rankid))
