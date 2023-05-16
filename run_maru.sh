#!/bin/bash

target="TStage"
out_path="ctrans-stage-1024T"

python -m marugoto.mil crossval \
    --clini_table "/mnt/SATELLITE_03/tim_warmup/TCGA-CRC-DX_CLINI.xlsx" \
    --slide-csv "/mnt/SATELLITE_03/tim_warmup/TCGA-CRC-DX_SLIDE.csv" \
    --feature-dir "/mnt/SATELLITE_03/tim_warmup/CACHE-TCGA-CRC-CTP" \
    --target-label ${target}  \
    --output-path /mnt/SATELLITE_03/tim_warmup/TCGA-CRC/${out_path} \
    --n-splits 5

true_labels=("T1" "T2" "T3" "T4")
for t in ${true_labels[@]}; do
    echo "Creating ROC plots for label ${t}"
    python -m marugoto.visualizations.roc \
                /mnt/SATELLITE_03/tim_warmup/TCGA-CRC/${out_path}/fold-*/patient-preds.csv \
                --outpath /mnt/SATELLITE_03/tim_warmup/TCGA-CRC/${out_path}/roc \
                --target-label ${target} --true-label ${t}
done
