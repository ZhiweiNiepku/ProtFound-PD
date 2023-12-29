# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
TopK for text generation
"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P


def topk_fun(logits, topk=5):
    """Get topk"""
    target_column = logits[0].tolist()
    sorted_array = [(k, v) for k, v in enumerate(target_column)]
    sorted_array.sort(key=lambda x: x[1], reverse=True)
    topk_array = sorted_array[:topk]
    index, value = zip(*topk_array)
    index = np.array([index])
    value = np.array([value])
    return value, index

map_dict = {'L': 5, 'S': 6, 'A': 7, 'G': 8, 'E': 9, 'V': 10, 'T': 11, 'R': 12, 'D': 13, 'I': 14,
            'P': 15, 'K': 16, 'N': 17, 'F': 18, 'Q': 19, 'Y': 20, 'H': 21, 'M': 22, 'C': 23, 'W': 24,
            'X': 1, 'B': 2, 'Z': 3, 'U': 4, 'O': 26, 'SOT': 25, 'SHT': 27, 'MED': 28, 'LON': 29, 'EOT': 0, 'PAD': 0}
def encode(seq):
    # return [map_dict[c.upper()] for c in seq]

    return [map_dict[i] for i in list(seq)]


def sort_list(probs, p_args):
    dict1 = {}
    probs_res = []
    p_args_res = []
    for i, b in enumerate(p_args):
        dict1[b] = probs[i]
    sort_dict = sorted(dict1.items(), key=lambda x: x[1], reverse=True)

    for key in sort_dict:
        p_args_res.append(key[0])
        probs_res.append(key[1])
    return probs_res, p_args_res


def sampler(log_probs_revised, top_p, top_k_num, org_seq, mask_index, use_pynative=False):
    """Convert the log_probs to probability"""
    if use_pynative:
        logits = P.Pow()(10, Tensor(log_probs_revised, mstype.float32))
    else:
        logits = np.power(10, np.array(log_probs_revised, np.float32))

        # If top_p is less than 1.0, use top_p sampling
    if top_p < 1.0:
        # Only consider the 5000 largest logits to reduce computation
        if use_pynative:
            sorted_logits, index = P.TopK(sorted=True)(logits, 5000)
            cumsum_logits = P.CumSum()(sorted_logits, 1)
            cumsum_logits = cumsum_logits.asnumpy()
            index = index.asnumpy()
            sorted_logits = sorted_logits.asnumpy()
        else:
            sorted_logits, index = topk_fun(logits, 5000)
            cumsum_logits = np.cumsum(sorted_logits, 1)
        cumsum_logits = cumsum_logits[0]
        index = index[0]
        sorted_logits = sorted_logits[0]
        top_p_num = sum(cumsum_logits > top_p)
        # In case the probability is smooth, the sum of 5000 largest probabilities are not large enough
        if top_p_num == 0:
            top_p_num = 5000
        # Get the corresponding probs and indices
        probs = sorted_logits[:top_p_num]
        p_args = index[:top_p_num]
        p = probs / sum(probs)
        # if top_p is set to 1.0, use top_k sampling
    else:
        # Get the corresponding probs and indices
        if use_pynative:
            probs, p_args = P.TopK(sorted=True)(logits, top_k_num)
            probs = probs.asnumpy()
            p_args = p_args.asnumpy()
        else:
            probs, p_args = topk_fun(logits, top_k_num)
            # print("probs", probs, "p_args", p_args)


        probs = probs[0]
        p_args = p_args[0]
        # print("yuan_p", probs, "yuan_p_args", p_args)
        org_seq = encode(org_seq)
        # print("org_seq", org_seq, "index", mask_index)
        for i, p in enumerate(p_args):
            # print(i)
            org_index = mask_index
            if p in [7, 8, 18, 24, 22, 23]:
                # print("agfwmc----down")
                probs[i] = probs[i] * 0.1
            # 极性氨基酸
            # print("jixinganjisuan")
            if org_seq[mask_index] in [8, 20, 17, 19, 6, 11, 23, 16, 12, 21, 13, 9]:
                if p in [16, 12, 21, 13, 9]:
                    probs[i] = probs[i] * 10
            # 疏水氨基酸
            # print("shushuianjisuan")
            if org_seq[mask_index] in [7, 10, 5, 14, 18, 24, 22, 15]:
                j = 0
                while j < 5 and mask_index > 0:
                    if org_seq[mask_index] in [7, 10, 5, 14, 18, 24, 22, 15]:
                        j += 1
                        mask_index -= 1
                    else:
                        break
                if j == 4:
                    if p in [7, 10, 5, 14, 18, 24, 22, 15]:
                        probs[i] = probs[i] * 0.1
                mask_index = org_index
                k = 0
                while k < 5 and mask_index < len(org_seq):
                    if org_seq[mask_index] in [7, 10, 5, 14, 18, 24, 22, 15]:
                        k += 1
                        mask_index += 1
                    else:
                        break
                if k == 4:
                    if p in [7, 10, 5, 14, 18, 24, 22, 15]:
                        probs[i] = probs[i] * 0.1
                mask_index = org_index
            # 正电/负电
            if org_seq[mask_index] in [6, 12, 21]:
                j = 0
                while j < 5 and mask_index > 0:
                    if org_seq[mask_index] in [6, 12, 21]:
                        j += 1
                        mask_index -= 1
                    else:
                        break
                if j == 4:
                    if p in [6, 12, 21]:
                        probs[i] = probs[i] * 0.1
                k = 0
                mask_index = org_index
                while k < 5 and mask_index < len(org_seq):
                    if org_seq[mask_index] in [6, 12, 21]:
                        k += 1
                        mask_index += 1
                    else:
                        break
                if k == 4:
                    if p in [6, 12, 21]:
                        probs[i] = probs[i] * 0.1
                mask_index = org_index
            if org_seq[mask_index] in [13, 9]:
                j = 0
                while j < 5 and mask_index > 0:
                    if org_seq[mask_index] in [13, 9]:
                        j += 1
                        mask_index -= 1
                    else:
                        break
                if j == 4:
                    if p in [13, 9]:
                        probs[i] = probs[i] * 0.1
                mask_index = org_index
                k = 0
                while k < 5 and mask_index < len(org_seq):
                    if org_seq[mask_index] in [13, 9]:
                        k += 1
                        mask_index += 1
                    else:
                        break
                if k == 4:
                    if p in [13, 9]:
                        probs[i] = probs[i] * 0.1
                mask_index = org_index
            if mask_index > 0 and mask_index < len(org_seq) * 0.25:
                if p == 15:
                    probs[i] = probs[i] * 0.1
            if mask_index > len(org_seq) * 0.40 and mask_index < len(org_seq) * 0.60:
                if p == 15:
                    probs[i] = probs[i] * 0.1
            if mask_index > len(org_seq) * 0.75 and mask_index < len(org_seq) * 0.100:
                if p == 15:
                    probs[i] = probs[i] * 0.1
            if mask_index > len(org_seq) * 0.25 and mask_index < len(org_seq) * 0.40:
                l = int(len(org_seq) * 0.25)
                r = int(len(org_seq) * 0.40)
                if p in org_seq[l:r]:
                    continue
                elif p == 15:
                    probs[i] = probs[i] * 1.1
            if mask_index > len(org_seq) * 0.60 and mask_index < len(org_seq) * 0.75:
                l = int(len(org_seq) * 0.60)
                r = int(len(org_seq) * 0.75)



                if p in org_seq[l:r]:
                    continue
                elif p == 15:
                    probs[i] = probs[i] * 1.1

            probs, p_args = sort_list(probs, p_args)

        # print(probs, p_args)

        # Avoid rounding error
        if sum(probs) == 0:
            probs = np.array([1 / top_k_num for _ in range(top_k_num)])
        p = probs / sum(probs)

    # print("p", probs, "p_args", p_args)
    return p, p_args


def generate(model, origin_inputs, config, cur_ss, org_seq, index):
    # print("cur_ss", cur_ss)
    """
    Text generation

    Inputs:
        model: the model for inferencing
        origin_inputs: the original inputs based on which the model will continue writing
        config: inference configurations

    Returns:
        outputs: the ids for the generated text
    """
    # Get configurations for inference
    frequency_penalty = config.frequency_penalty
    presence_penalty = config.presence_penalty
    top_p = config.top_p
    top_k_num = config.top_k_num
    max_generate_length = config.max_generate_length
    seq_length = config.seq_length
    end_token = config.end_token
    use_pynative = config.use_pynative_op

    _, valid_length = origin_inputs.shape
    # If target length exceeds seq_length, use seq_length instead
    target_length = valid_length + max_generate_length
    target_length = seq_length if target_length > seq_length else target_length

    # A list of the frequency of each token
    frequency_list = np.array([[0 for _ in range(config.vocab_size)]])
    #frequency_list = np.array([[0 for _ in range(512)]])
    pad_length = seq_length - origin_inputs.shape[-1]
    # Pad original inputs to seq_length
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, 0))
    #print("input_ids is ", input_ids)

    # A single loop generates one token, loop until reaching target seq_length or generating eod token
    while valid_length < target_length:
        # print("===========================================")
        inputs = Tensor(input_ids, mstype.int32)
        # Indicate the exact token position
        current_index = valid_length - 1 if valid_length - 1 > 0 else 0
        current_index = Tensor([current_index], mstype.int32)
        # Call a single inference
        log_probs = model.predict(inputs, current_index)
        # Get the revised log_probs considering frequency and presence penalty to eliminate duplicate in generated results
        log_probs = log_probs.asnumpy().reshape(1, config.vocab_size)
        #log_probs = log_probs.asnumpy().reshape(1, -1)
        log_probs_revised = log_probs - frequency_list * frequency_penalty - (frequency_list > 0) * presence_penalty

        p, p_args = sampler(log_probs_revised, top_p, top_k_num, org_seq, index, use_pynative)
        p = p[0:5]
        p = [pro / sum(p) for pro in p]
        # Random select a token as final output for this round
        target_index = np.random.choice(5, p=p)
        # Stop judgment

        res = 999
        list1 = [7, 10, 5, 14, 18, 24, 22, 15]
        list2 = [21, 16, 12]
        list3 = [8, 23, 19, 17, 6, 20, 11]
        list4 = [9, 13]

        flag = 0

        if p_args[target_index] == end_token or valid_length == target_length - 1:
            outputs = input_ids
            break

        # update frequency list

        if cur_ss in list1:
            flag = 1
        elif cur_ss in list2:
            flag = 2
        elif cur_ss in list3:
            flag = 3
        elif cur_ss in list4:
            flag = 4
        # print("--------------------------------------------")
        # print("p_args[target_index]", p_args[target_index])
        # print("p_args", p_args)
        # print("flag", flag)
        for ss in p_args:
            if ss not in [2, 3, 4, 26]:
            # print(ss)
                if flag == 1 and ss in list1:
                    res = ss
                    break
                if flag == 2 and ss in list2:
                    res = ss
                    break
                if flag == 3 and ss in list3:
                    res = ss
                    break
                if flag == 4 and ss in list4:
                    res = ss
                    break
        # 不满足同源条件不突变
        if res == 999:
            res = cur_ss
        # print("res", res)
        target = p_args[target_index]
        frequency_list[0][target] = frequency_list[0][target] + 1
        # Modify input_ids with newly generated token
        input_ids[0][valid_length] = res
        # print("res1", input_ids[0][valid_length])
        valid_length += 1

        break
    outputs = input_ids
    # Return valid outputs out of padded outputs
    length = np.sum(outputs != 0)
    outputs = outputs[0][:length]
    # print("outputs", outputs)
    return outputs


def generate_increment(model, origin_inputs, config, org_seq, index):
    """
    Text generation for incremental inference

    Inputs:
        model: the model for inferencing
        origin_inputs: the original inputs based on which the model will continue writing
        config: inference configurations

    Returns:
        outputs: the ids for the generated text
    """
    # Get configurations for inference
    frequency_penalty = config.frequency_penalty
    presence_penalty = config.presence_penalty
    top_p = config.top_p
    top_k_num = config.top_k_num
    max_generate_length = config.max_generate_length
    seq_length = config.seq_length
    end_token = config.end_token
    use_pynative = config.use_pynative_op

    _, valid_length = origin_inputs.shape
    # Init outputs with original inputs
    outputs = [origin_inputs[0][i] for i in range(valid_length)]
    # If target length exceeds seq_length, use seq_length instead
    target_length = valid_length + max_generate_length
    target_length = seq_length if target_length > seq_length else target_length

    # A list of the frequency of each token
    frequency_list = np.array([[0 for _ in range(config.vocab_size)]])
    pad_length = seq_length - origin_inputs.shape[-1]
    # Pad original inputs to seq_length
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, 0))
    print("input_ids is ", input_ids)

    # Indicate the exact token position
    current_index = valid_length - 1 if valid_length - 1 > 0 else 0
    batch_valid_length = Tensor(np.array([current_index]), mstype.int32)
    current_index = Tensor(np.array([current_index]), mstype.int32)
    # For first graph, not_init should be false
    init_true = Tensor([True], mstype.bool_)
    init_false = Tensor([False], mstype.bool_)
    init = init_false
    # Claim the first graph
    model.predict_network.add_flags_recursive(is_first_iteration=True)
    # Call a single inference with input size of (bs, seq_length)
    logits = model.predict(Tensor(input_ids, mstype.int32), current_index, init, batch_valid_length)

    # Claim the second graph and set not_init to true
    init = init_true
    model.predict_network.add_flags_recursive(is_first_iteration=False)

    # A single loop generates one token, loop until reaching target seq_length or generating eod token
    while valid_length < target_length:
        # Reshape the output logits
        logits = logits.asnumpy()
        log_probs = logits.reshape(1, config.vocab_size)

        # Get the revised log_probs considering frequency and presence penalty to eliminate duplicate in generated results
        log_probs_revised = log_probs - frequency_list * frequency_penalty - (frequency_list > 0) * presence_penalty

        p, p_args = sampler(log_probs_revised, top_p, top_k_num, org_seq, index, use_pynative)
        # Random select a token as final output for this round
        target_index = np.random.choice(len(p), p=p)
        # Stop judgment
        if p_args[target_index] == end_token or valid_length == target_length - 1:
            break

        # Update frequency list
        target = p_args[target_index]
        frequency_list[0][target] = frequency_list[0][target] + 1
        valid_length += 1

        batch_valid_length = Tensor(np.array([valid_length - 1]), mstype.int32)
        current_index = Tensor(np.array([0]), mstype.int32)
        input_id = Tensor([[target]], mstype.int32)
        # Update outputs with current generated token
        outputs.append(int(target))

        # Call a single inference with input size of (bs, 1)
        logits = model.predict(input_id, current_index, init, batch_valid_length)
    # Return valid outputs out of padded outputs
    return np.array(outputs)
