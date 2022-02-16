m=dense_small
mode=extra_linear
p=self
MODELS_PATH=/mnt/ssd-1/igor/gpt-neox/models
ROOT=~/igor/gpt-neox

for j in {1..12}; do
	for i in {0..12}; do
		xm=$m.$mode.${p}_lens_$j.$i
		cd $MODELS_PATH
		$ROOT/tools/pythia/model_xform.py --predict $p --mode $mode --num_layers $i --head $m.$mode.$p.$j $m $xm || break
		cd -
		./tools/w.sh $xm || break
		rm -rf $MODELS_PATH/$xm
	done
done


