#!/bin/bash
cd /lhome/ext/uib107/uib107c/repos/E2EPansharpening
python src/train_ours.py --model VPSNetMalisat --upsampling_type malisat --postprocessing_type spectral --nickname PruebaMalisatWVLimit --loss_function L1 --dataset worldview --stages 3 --epochs 2000 --batch_size 16 --limit_dataset 520 