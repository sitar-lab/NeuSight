import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForPreTraining,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)
from transformers.utils.fx import (
    symbolic_trace as symbolic_trace_transformers,
)
from .analysis import NodeProp, visualize
import pandas as pd
import uuid
import json
import gc
import torch

# setup for moe
import transformers.utils.fx
transformers.utils.fx.check_if_model_is_supported = lambda x : True
import transformers.models.switch_transformers.modeling_switch_transformers

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False

active = 25
sample = 5

expert_capacity_ = None
num_experts_ = None

def measure_cuda_kernel(kernel, args, kwargs, measure_time=True):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latency_list = []

    new_args = []
    for a in args:
        if hasattr(a, "cuda"):
            new_args.append(a.cuda())
        else:
            new_args.append(a)

    result = kernel(*new_args, **kwargs)
    if not measure_time:
        return result, 0

    for _ in range(active):
        start.record()
        result = None
        result = kernel(*new_args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        latency_list.append(start.elapsed_time(end))

    latency_list.sort()
    latency_list = latency_list[:sample]
    latency = sum(latency_list)/len(latency_list)

    return result, latency

def set_moe():
    # set_moe.expert_capacity = expert_capacity
    # set_moe.num_experts = num_experts

    def custom_load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
        num_experts = router_probs.shape[-1]

        # cast the expert indices to int64, otherwise one-hot encoding will fail
        if expert_indices.dtype != torch.int64:
            expert_indices = expert_indices.to(torch.int64)

        if len(expert_indices.shape) == 2:
            expert_indices = expert_indices.unsqueeze(2)

        expert_mask = torch.nn.functional.one_hot(expert_indices, set_moe.num_experts) 

        # For a given token, determine if it was routed to a given expert.
        expert_mask = torch.max(expert_mask, axis=-2).values

        # cast to float32 otherwise mean will fail
        expert_mask = expert_mask.to(torch.float32)
        tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

        router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
        return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (set_moe.num_experts**2)

    def custom_switchtransmlp_forward(self, hidden_states):
        router_mask, router_probs, router_logits = self.router(hidden_states)
        expert_index = torch.argmax(router_mask, dim=-1)

        next_states = hidden_states.clone()
        assert(hidden_states.dtype == next_states.dtype)
        assert(hidden_states.device == next_states.device)
        batch, seqlen, hid = hidden_states.shape

        hidden_states_padded = torch.zeros(batch*set_moe.expert_capacity, hid, dtype=hidden_states.dtype, device=hidden_states.device)
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, :, idx].bool()
            num_chosen  = torch.count_nonzero(token_indices)
            hidden_states_padded[:num_chosen,:] = hidden_states[token_indices]
            hidden_states_padded = expert(hidden_states_padded)
            # hidden_states_padded = hidden_states_padded.to(next_states.dtype)
            next_states[token_indices] = hidden_states_padded[:num_chosen,:]

        hidden_states = router_probs * next_states
        
        return hidden_states, (router_logits, expert_index)

    transformers.models.switch_transformers.modeling_switch_transformers.load_balancing_loss_func = custom_load_balancing_loss_func
    transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersSparseMLP.forward = custom_switchtransmlp_forward

def get_model(model_config_path, is_train, device, fusion):

    with open(model_config_path) as f:
        config_json = json.load(f)
    model_name = config_json["architectures"][0].lower()

    if "gpt" in model_name:
        n_layer = config_json["n_layer"]
    elif "switch" in model_name:
        set_moe.expert_capacity = config_json["expert_capacity"]
        set_moe.num_experts = config_json["num_experts"]
        n_layer = 0
    else:
        n_layer = config_json["num_hidden_layers"]

    config = AutoConfig.from_pretrained(model_config_path)


    if is_train:
        if "bert" in model_name.lower():
            model = AutoModelForPreTraining.from_config(config, attn_implementation="eager")
        elif "gpt" in model_name.lower():
            model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        elif "opt" in model_name.lower():
            model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        elif "switch" in model_name.lower():
            model = AutoModelForSeq2SeqLM.from_config(config, attn_implementation="eager")
        model = model.train()
    else:
        if "bert" in model_name.lower():
            model = AutoModelForSequenceClassification.from_config(config, attn_implementation="eager")
        elif "gpt" in model_name.lower():
            model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        elif "opt" in model_name.lower():
            model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        elif "switch" in model_name.lower():
            model = AutoModelForSeq2SeqLM.from_config(config, attn_implementation="eager")
        else:
            NotImplementedError()
        model = model.eval()

    # print("param count : ", sum(p.numel() for p in model.parameters()))

    if fusion:
        model = torch.compile(model)

    model = model.to(device)

    return model, n_layer

def trace_fx_graph(batch_size, sequence_length, model_config_path, is_train, bench, single_layer, fusion):

    gc.collect()
    torch.cuda.empty_cache()

    with open(model_config_path) as f:
        config_json = json.load(f)
    model_name = config_json["architectures"][0].lower()

    # change number of layers to 1
    if single_layer:
        if "gpt" in model_name:
            config_json["n_layer"] = 1
        else:
            config_json["num_hidden_layers"] = 1

    tmp_fname = f"/tmp/{uuid.uuid4().hex}.json"
    with open(tmp_fname, "w") as f:
        json.dump(config_json, f)

    try:
        if bench:
            device = "cuda"
        else:
            device = "cpu"
        model, n_layer = get_model(model_config_path=tmp_fname, is_train=is_train, device=device, fusion=False) # trace non-fused model instead
        # print("model loaded", flush=True)

        set_moe()
        # print("moe set", flush=True)

        with torch.no_grad():
            import inspect
            sig = inspect.signature(model.forward)
            # print(sig)
            graphmodule: torch.fx.GraphModule = symbolic_trace_transformers(model)
        # print("graph traced", flush=True)

        node_dict = []
        for n in graphmodule.graph.nodes:
            nd = n.__dict__.copy()
            nd["Name"] = n.name
            node_dict.append(nd)
        df = pd.DataFrame(node_dict)
        if 'graph' in df.columns:
            df = df.drop("graph", axis=1)

        nodeprop = NodeProp(graphmodule)
        if "switch" in model_name: # 
            input_ids = torch.ones(batch_size, sequence_length, dtype=torch.int64, device=device)
            decoder_input_ids = input_ids
            decoder_attention_mask = input_ids
            inputs = [input_ids, decoder_input_ids, decoder_attention_mask]
        else:
            input_ids = torch.ones(batch_size, sequence_length, dtype=torch.int64, device=device)
            inputs = [input_ids]
        graphmodule = nodeprop.propagate(*inputs, backward=is_train, bench=bench)
        print("node proped", flush=True)
    except Exception as e:
        raise e
        print(e)
        return None
    
    # visualize(graphmodule.graph)

    node_dict = []
    for n in graphmodule.graph.nodes:
        nd = n.__dict__.copy()
        nd["Name"] = n.name
        nd["meta"] = n.meta
        nd["_input_nodes"] = str([m.name for m in n.all_input_nodes])
        nd["users"] = str([u for u in n.users])
        node_dict.append(nd)
    df = pd.DataFrame(node_dict)
    if 'graph' in df.columns:
        df = df.drop("graph", axis=1)
    # df.to_csv(out_name, index=False)

    return df

def measure_e2e(batch_size, sequence_length, model_config_path, is_train, fusion):
    gc.collect()
    torch.cuda.empty_cache()

    with open(model_config_path) as f:
        config_json = json.load(f)
    model_name = config_json["architectures"][0].lower()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    forward_list = []
    backward_list = []

    try:
        model, n_layer = get_model(model_config_path=model_config_path, is_train=is_train, device="cuda", fusion=fusion)
        # print("memory allocated after get model : ", torch.cuda.memory_allocated(device=None) / 1024 / 1024 / 1024)
        
        input_ids = torch.ones(
            batch_size, sequence_length, dtype=torch.int64, device="cuda")

        if "switch" in model_name: # feed label too for switch transformer
            input_ids = torch.ones(batch_size, sequence_length, dtype=torch.int64, device="cuda")
            decoder_input_ids = input_ids
            decoder_attention_mask = input_ids
            inputs = [input_ids, decoder_input_ids, decoder_attention_mask]
        else:
            input_ids = torch.ones(batch_size, sequence_length, dtype=torch.int64, device="cuda")
            inputs = [input_ids]

        for _ in range(active):
            gc.collect()
            torch.cuda.empty_cache()
            
            if is_train:
                # forwrad measure
                start.record()
                result = model(*inputs)
                end.record()
                torch.cuda.synchronize()
                forward = start.elapsed_time(end)
                forward_list.append(forward)
    
                # compute loss
                if hasattr(result, "prediction_logits"):
                    logits = result["prediction_logits"]
                else:
                    logits = result["logits"]
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
                result = None
                logits = None

                # backward measure
                start.record()
                loss.backward()
                end.record()
                loss = None
                torch.cuda.synchronize()
                backward = start.elapsed_time(end)
                backward_list.append(backward)
            else:
                # forwrad measure with no grad
                with torch.no_grad():
                    start.record()
                    result = model(*inputs)
                    end.record()
                    torch.cuda.synchronize()
                    forward = start.elapsed_time(end)
                    forward_list.append(forward)
                    result = None
        
    except Exception as e:
        if "out of memory" in str(e).lower():
            print("e2e measure out of memory")
            return None
        else:
            raise e
    forward_list.sort()
    forward_list = forward_list[:sample]
    forward = sum(forward_list)/len(forward_list)

    # backwards were in oom or no grad
    if is_train:
        backward_list.sort()
        backward_list = backward_list[:sample]
        backward = sum(backward_list)/len(backward_list)
    else:
        backward = 0

    # e2e latency
    e2e = forward + backward


    # Sample dictionary
    e2e_dict = {
        "num_layer": n_layer,
        "e2e_latency": e2e,
        "fw_latency": forward,
        "bwall_latency": backward
    }

    return e2e_dict

def trace_graph(
            model_config_path, 
            sequence_length, 
            batch_size, 
            is_train, 
            bench, 
            single_layer=True, 
            fusion=False,
            distributed=False,
            dp_degree=1,
            pp_degree=1,
            pp_num_microbatch=1,
            tp_degree=1,
):
    # print(model_config_path, batch_size, sequence_length, "is_train ", is_train, "single_layer ", single_layer, "fusion ", fusion)

    if bench:
        # measure training e2e latency
        e2e_entry = measure_e2e(batch_size=batch_size, sequence_length=sequence_length, model_config_path=model_config_path, is_train=is_train, fusion=fusion)
        gc.collect()
        torch.cuda.empty_cache()
        # print("memory allocated after e2e measurement : ", torch.cuda.memory_allocated(device=None) / 1024 / 1024 / 1024)
        if e2e_entry is None:
            return None, None

        # trace operator graph
        df = trace_fx_graph(batch_size=batch_size, sequence_length=sequence_length, model_config_path=model_config_path, is_train=is_train, bench=True, single_layer=single_layer, fusion=fusion)
        gc.collect()
        torch.cuda.empty_cache()
        # df = pd.concat([df, pd.DataFrame([e2e_entry])])

    else:
        # no bench
        e2e_entry = None
        
        # trace operator graph
        df = trace_fx_graph(batch_size=batch_size, sequence_length=sequence_length, model_config_path=model_config_path, is_train=is_train, bench=False, single_layer=single_layer, fusion=fusion)

    return df, e2e_entry

