#!/bin/bash
# Download MS-COCO 2017 images and captions.
# Run from the project root: bash download_coco.sh

set -e

COCO_ROOT="./data/coco"
mkdir -p "$COCO_ROOT"
cd "$COCO_ROOT"

echo "=== Downloading COCO 2017 annotations ==="
if [ ! -d "annotations" ]; then
    wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
    echo "Annotations downloaded."
else
    echo "Annotations already exist, skipping."
fi

echo "=== Downloading COCO 2017 train images (~18GB) ==="
if [ ! -d "train2017" ]; then
    wget -q http://images.cocodataset.org/zips/train2017.zip
    unzip -q train2017.zip
    rm train2017.zip
    echo "Train images downloaded."
else
    echo "Train images already exist, skipping."
fi

echo "=== Downloading COCO 2017 val images (~1GB) ==="
if [ ! -d "val2017" ]; then
    wget -q http://images.cocodataset.org/zips/val2017.zip
    unzip -q val2017.zip
    rm val2017.zip
    echo "Val images downloaded."
else
    echo "Val images already exist, skipping."
fi

echo "=== Done! Data at $COCO_ROOT ==="
