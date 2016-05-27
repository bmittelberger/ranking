#!/bin/bash
CVals=(0.125 0.25 0.5 1 2 4 8)
GammaVals=(0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5)
for i in "${CVals[@]}"
do
	for j in "${GammaVals[@]}"
	do
		echo "C = $i Gamma = $j"
		sh run_w_hyperparams.sh ../data/pa3.signal.train ../data/pa3.rel.train ../data/pa3.signal.dev ../data/pa3.rel.dev ../data/idfs 2 $i $j
	done
done
