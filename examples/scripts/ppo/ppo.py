# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
#     "trackio",
# ]
# ///

import os
import shutil
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:18669'  # 将端口号替换成你的实际端口
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:18669' # 将端口号替换成你的实际端口
import torch
from accelerate import PartialState
from datasets import load_dataset,load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

"""

python -i examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir pythia-1b-deduped-descriptiveness-sentiment-trl-style-ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir pythia-1b-deduped-descriptiveness-sentiment-trl-style-ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0
    
lgx:
python examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0
"""


def set_args():
    script_args = ScriptArguments()
    script_args.dataset_name = "/Users/guoxing.lan/projects/dataset/for_rl/descriptiveness-sentiment-trl-style"
    script_args.dataset_train_split = "descriptiveness"
    # num_ppo_epochs和num_mini_batches都是为了平衡效率与稳定性而设计的“数据复用”策略。跟off-policy里面的“经验回放”不是一回事
    training_args = PPOConfig(
        learning_rate=3e-6,
        num_ppo_epochs=1,  # 因为roll_out成本很大。在每次收集完一批经验数据（rollout）后，使用该数据对模型进行参数更新的轮数
        num_mini_batches=1,  # 将一次经验收集（rollout）得到的数据批（batch）进一步划分成更小的“小批次”（minibatches）用于梯度计算。1说明不划分
        output_dir=r"models\minimal\ppo",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        local_rollout_forward_batch_size=2,  # 为了解决大模型在生成阶段显存不足的问题而引入的参数。它会将一个大的生成任务拆分成更小的批次依次执行，然后再合并结果。
        total_episodes=100,  # 处理的总样本数。num_train_epochs = total_episodes / train_dataset_len
        sft_model_path=r'D:\projects\models\EleutherAI\pythia-160m',
        reward_model_path=r'D:\projects\models\EleutherAI\pythia-160m',
        missing_eos_penalty=1.0,
        no_cuda=True,
        use_cpu=True
    )  # PPOConfig<-OnPolicyConfig<-TrainingArguments

    model_args = ModelConfig(model_name_or_path="/Users/guoxing.lan/projects/models/EleutherAI/pythia-160m",
                             )
    return script_args, training_args, model_args


if __name__ == "__main__":
    # parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    # script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    script_args, training_args, model_args = set_args()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    # dataset = load_dataset(
    #     script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split
    # )
    # dataset0 = load_dataset(
    #     "trl-internal-testing/descriptiveness-sentiment-trl-style", name=script_args.dataset_config, split=script_args.dataset_train_split
    # )
    dataset = load_from_disk(
        script_args.dataset_name
    )
    dataset=dataset[script_args.dataset_train_split]
    dataset=dataset.select(range(0,60))
    eval_samples = 10
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    dataset_text_field = "prompt"


    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )


    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        # if True:
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    ################
    # Training
    ################
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()
