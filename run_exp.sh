#!/bin/bash
set -eux
for i in {1..5}; do
    timestamp=$(date +%Y%m%d-%H%M%S)
    exp_name="vit_lstm_feat_resnet50_nopretrain_$timestamp"
    clini_table="$HOME/dataset/duke_marugoto/clinical_table.csv"
    slide_csv="$HOME/dataset/duke_marugoto/slide_table.csv"
    feature_dir="$HOME/dataset/duke_marugoto/features_odelia_sub_resnet50_nopretrain/train_val"
    deploy_dir="$HOME/dataset/duke_marugoto/features_odelia_sub_resnet50_nopretrain/test/"
    target_label="Malign"
    output_path="./results/$exp_name"

    python -m marugoto.mil train \
            --clini-table "$clini_table" \
            --slide-csv "$slide_csv" \
            --feature-dir "$feature_dir" \
            --target-label "$target_label" \
            --output-path "$output_path"

    python -m marugoto.mil deploy \
            --clini-table "$clini_table" \
            --slide-csv "$slide_csv" \
            --feature-dir "$deploy_dir" \
            --target-label "$target_label" \
            --model-path "$output_path/export.pkl" \
            --output-path "$output_path"


    python -m marugoto.stats.categorical \
            "$output_path/patient-preds.csv" \
            --outpath "$output_path" \
            --target_label "$target_label"

    python -m marugoto.visualizations.roc \
            "$output_path/patient-preds.csv" \
            --outpath "$output_path" \
            --target-label "$target_label" \
            --true-label 1

    python -m marugoto.visualizations.prc \
            "$output_path/patient-preds.csv" \
            --outpath "$output_path" \
            --target-label "$target_label" \
            --true-label 1

    echo "Run $i finished."
done