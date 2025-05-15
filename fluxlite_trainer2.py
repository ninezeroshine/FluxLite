# fluxlite_trainer.py
"""
Минималистичный CLI для тренировки LoRA/FluxGYM моделей на основе sd-scripts
Версия 0.1 (MVP) — только CLI, только одна команда: train
"""
import os
import subprocess
import sys
import yaml

CONFIG_FILE = 'fluxlite_config.yaml'

DEFAULT_CONFIG = {
    'project': 'icon-test-2',
    'dataset_dir': '/workspace/fluxlite/datasets/icon-test-2',
    'output_dir': '/workspace/fluxlite/outputs/icon-test-2',
    'pretrained_model': '/workspace/fluxlite/models/unet/flux1-dev2pro.safetensors',
    'clip_l': '/workspace/fluxlite/models/clip/clip_l.safetensors',
    't5xxl': '/workspace/fluxlite/models/clip/t5xxl_fp16.safetensors',
    'vae': '/workspace/fluxlite/models/vae/ae.sft',
    'resolution': 1024,
    'batch_size': 1,
    'max_train_epochs': 100,
    'save_every_n_epochs': 10,
    'learning_rate': 1e-5,
    'network_dim': 32,
    'seed': 42,
}

TEMPLATE_TRAIN_SH = '''\
#!/bin/bash
source ../env/bin/activate

accelerate launch \
  --mixed_precision bf16 \
  sd-scripts/flux_train_network.py \
  --pretrained_model_name_or_path {pretrained_model} \
  --clip_l {clip_l} \
  --t5xxl {t5xxl} \
  --ae {vae} \
  --cache_latents_to_disk \
  --save_model_as safetensors \
  --sdpa --persistent_data_loader_workers \
  --max_data_loader_n_workers 1 \
  --seed {seed} \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --save_precision bf16 \
  --network_module networks.lora_flux \
  --network_dim {network_dim} \
  --optimizer_type adamw8bit \
  --learning_rate {learning_rate} \
  --cache_text_encoder_outputs \
  --cache_text_encoder_outputs_to_disk \
  --fp8_base \
  --highvram \
  --max_train_epochs {max_train_epochs} \
  --save_every_n_epochs {save_every_n_epochs} \
  --dataset_config {output_dir}/dataset.toml \
  --output_dir {output_dir} \
  --output_name {project} \
  --timestep_sampling shift \
  --discrete_flow_shift 3.1582 \
  --model_prediction_type raw \
  --guidance_scale 1 \
  --loss_type l2 \
  {resume_line} \
  "$@"
'''

def init_config():
    if os.path.exists(CONFIG_FILE):
        print(f"[!] {CONFIG_FILE} уже существует. Проверь настройки перед запуском.")
        return
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(DEFAULT_CONFIG, f)
    print(f"[+] {CONFIG_FILE} создан. Отредактируй его и запусти снова.")

def gen_train_sh(cfg):
    path = os.path.join(cfg['output_dir'], 'train.sh')
    os.makedirs(cfg['output_dir'], exist_ok=True)
    with open(path, 'w') as f:
        f.write(TEMPLATE_TRAIN_SH.format(**cfg))
    os.chmod(path, 0o755)
    print(f"[+] Сгенерирован {path}")
    return path

def main():
    if len(sys.argv) == 1 or sys.argv[1] in ('--help', '-h'):
        print("""
FluxLite Trainer CLI
===================
  fluxlite_trainer.py init      # создать базовый fluxlite_config.yaml
  fluxlite_trainer.py train     # сгенерить train.sh и запустить
""")
        return
    cmd = sys.argv[1]
    if cmd == 'init':
        init_config()
        return
    if cmd == 'train':
        if not os.path.exists(CONFIG_FILE):
            print(f"[!] Нет {CONFIG_FILE}. Запусти `python fluxlite_trainer.py init`")
            return
        with open(CONFIG_FILE) as f:
            cfg = yaml.safe_load(f)
        sh = gen_train_sh(cfg)
        print(f"[~] Запуск тренировки... (Ctrl+C для остановки)")
        subprocess.call([sh])
        return
    print(f"[?] Неизвестная команда: {cmd}")

if __name__ == '__main__':
    main()
