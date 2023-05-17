#!/bin/bash

target="TStage"
out_path="TCGA-results/pos-enc/ctrans-stage"

python -m marugoto.mil crossval \
    --clini_table "/scratch/ws/1/s1787956-tim-cpath/CACHE-TCGA-CRC/TCGA-CRC-DX_CLINI.xlsx" \
    --slide-csv "/scratch/ws/1/s1787956-tim-cpath/CACHE-TCGA-CRC/TCGA-CRC-DX_SLIDE.csv" \
    --feature-dir "/scratch/ws/1/s1787956-tim-cpath/CACHE-TCGA-CRC-CTP" \
    --target-label ${target}  \
    --output-path /scratch/ws/1/s1787956-tim-cpath/${out_path} \
    --n-splits 5

true_labels=("T1" "T2" "T3" "T4")
for t in ${true_labels[@]}; do
    echo "Creating ROC plots for label ${t}"
    python -m marugoto.visualizations.roc \
                /scratch/ws/1/s1787956-tim-cpath/${out_path}/fold-*/patient-preds.csv \
                --outpath /scratch/ws/1/s1787956-tim-cpath/${out_path}/roc \
                --target-label ${target} --true-label ${t}
    done
