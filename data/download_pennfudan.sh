#!/usr/bin/env bash
set -e

mkdir -p data
cd data

wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
unzip -q PennFudanPed.zip
rm PennFudanPed.zip
echo "Done. Dataset extracted to: data/PennFudanPed"
