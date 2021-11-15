#!/usr/bin/env bash
#bash
#conda activate torchpy36
# python trainer.py
# python -m paddle.distributed.launch --selected_gpus '0,1' trainer.py
python -m paddle.distributed.launch --selected_gpus '0' trainer.py