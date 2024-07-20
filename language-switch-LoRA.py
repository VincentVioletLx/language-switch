import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    TextStreamer,
    Trainer,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer
import os, wandb
"""
bitsandbytes: 专为量化设计的库 用于减少LLM在GPU上的内存占用
peft: 用于将LoRA适配器集成到LLM上
trl: 包含一个SFT类 用于辅助微调模型
accelerate和xformer: 提高模型推理速度 优化性能
wandb: 用于跟踪和观察训练过程
datasets: 与HuggingFace一起使用 加载模型和数据集
"""

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_name = "/home/nfs02/model/Qwen1.5-1.8B"
data_files = {
    "train": "/home/nfs02/liux/LLaMA-Factory-main/data/language_switch.json",
    "test": "/home/nfs02/liux/LLaMA-Factory-main/data/language_switch_eval.json"
}

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 模型将以4位量化格式加载
    bnb_4bit_quant_type="nf4",  # 指定四位量化类型为nf4
    bnb_4bit_compute_dtype=torch.float16,  # 指定计算数据类型为torch.float16
    bnb_4bit_use_double_quant=False,  # 不使用双重量化
)

# 加载模型 分词器和数据集
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=bnb_config)

model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True # 在生成序列时自动添加结束标记
tokenizer.padding_side = "left" # 填充位置

dataset = load_dataset('json', data_files=data_files)

def preprocess_function(examples):
    return {
        'text': [f"{instr}\n{inp}" for instr, inp in zip(examples['instruction'], examples['input'])],
        'label': examples['output']
    }

dataset = dataset.map(preprocess_function, batched=True, remove_columns=['instruction', 'input', 'output'])

# wandb配置
wandb.login(key="b65ce85cbf05c06f0e7ecc89405ca4d919b7869c")
run = wandb.init(project="finetune Qwen1.5-1.8B", job_type="training")

# 计算训练参数量
def print_trainable_parameters(model):
    trainable_param = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_param += param.numel()
    print(f"训练参数量: {trainable_param} || 总参数量: {all_param} || 训练参数量占比%: {100 * (trainable_param / all_param):.2f}%")

# LoRA配置
peft_config = LoraConfig(
    r = 8,
    lora_alpha = 16, # 一般设置为α=2r
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"]
)

# 超参数配置
training_arguments = TrainingArguments(
    output_dir = "/home/nfs02/liux/checkpoint-output",
    num_train_epochs = 5,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4, # 梯度累计步数
    # optim = "adamw_torch",
    save_strategy = "steps",
    save_steps = 100, # 每100步保存一次模型
    logging_steps = 30, 
    learning_rate = 5e-5,
    weight_decay = 0.001, # 权重衰减系数 用于L2正则化 防止过拟合
    fp16 = True,
    max_grad_norm = 1, # 最大梯度范数 用于梯度裁剪
    max_steps = -1, # 最大训练步数 -1表示无限制
    warmup_ratio = 0.3, # 学习率预热比例
    group_by_length = True, # 将长度相近的序列分组以加速训练
    lr_scheduler_type = "linear", # 学习率调度器类型
    report_to = "wandb", # 报告到wandb
)

# SFT超参数
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset['train'],
    peft_config = peft_config,
    tokenizer = tokenizer,
    dataset_text_field = "text",
    args = training_arguments,
    packing = False
)

# 开始训练
trainer.train()

# 计算训练参数量
model = get_peft_model(model, peft_config)
print(print_trainable_parameters(model))

trainer.model.save_pretrained("/home/nfs02/liux/Qwen1.5-1.8B-lora0720")
wandb.finish()

# 模型合并
def apply_lora(base_model_path, lora_model_path, merged_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    new_model = PeftModel.from_pretrained(base_model, lora_model_path)
    model = new_model.merge_and_upload()

    model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)

apply_lora("/home/nfs02/liux/Qwen1.5-1.8B", "/home/nfs02/liux/Qwen1.5-1.8B-lora0720", "/home/nfs02/liux/Qwen1.5-1.8B-lora0720-merged")
