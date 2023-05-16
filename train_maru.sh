#!/bin/bash

epochs=({2..6})
targets=("isMSIH" "BRAF" "KRAS")
true_labels=("MSIH" "MUT" "MUT")

for e in ${epochs[@]}; do
    echo "Starting training for epoch ${e}"
    for i in ${!targets[@]}; do
        echo "Training on target ${targets[$i]}"
        python -m marugoto.mil crossval  \
            --clini_table "/mnt/SATELLITE_03/tim_warmup/TCGA-CRC-DX_CLINI.xlsx"    \
            --slide-csv "/mnt/SATELLITE_03/tim_warmup/TCGA-CRC-DX_SLIDE.csv"   \
            --feature-dir /mnt/SATELLITE_03/tim_warmup/output-tcga-swin-e${e}/E2E_macenko_swin-epoch${e} \
            --target-label ${targets[$i]} \
            --output-path /mnt/SATELLITE_03/tim_warmup/TCGA-CRC/swin-e${e}/${targets[$i]}/ \
            --n-splits 5
        
        python -m marugoto.visualizations.roc \
                /mnt/SATELLITE_03/tim_warmup/TCGA-CRC/swin-e${e}/${targets[$i]}/fold-*/patient-preds.csv \
                --outpath /mnt/SATELLITE_03/tim_warmup/TCGA-CRC/swin-e${e}/${targets[$i]}/roc \
                --target-label ${targets[$i]} --true-label ${true_labels[$i]}
        done
    echo "Done with epoch ${e}!"
    done
