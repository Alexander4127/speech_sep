#!/bin/bash

printf "Preparing custom dataset...\n"

printf "Copying mixed audio...\n"
mkdir -p data/datasets/mixed/data/custom/mix
cp data/datasets/mixed/data/test-clean/*-mixed.wav data/datasets/mixed/data/custom/mix || exit 1

printf "Copying target audio...\n"
mkdir -p data/datasets/mixed/data/custom/target
cp data/datasets/mixed/data/test-clean/*-target.wav data/datasets/mixed/data/custom/target

printf "Copying ref audio...\n"
mkdir -p data/datasets/mixed/data/custom/ref
cp data/datasets/mixed/data/test-clean/*-ref.wav data/datasets/mixed/data/custom/ref
