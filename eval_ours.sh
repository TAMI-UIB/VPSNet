#!/bin/bash

weights=weights_best.pth

while read -r path_to_model; do
    path=$path_to_model$weights
    IFS='/'
    read -ra parts <<< "$path_to_model"
    model="${parts[8]}"
    nickname="${parts[9]}"
    IFS=' '
    python src/eval_ours.py --dataset worldview --model $model --nickname $nickname --model_path $path
done < model_paths/ours_models_path.txt
