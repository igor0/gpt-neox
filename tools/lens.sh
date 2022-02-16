m=dense_medium
mode=extra_linear
p=self
MODELS_PATH=/mnt/ssd-1/igor/gpt-neox/models
ROOT=~/igor/gpt-neox

for j in 18 24 3 9 15 21; do
	for i in {0..24}; do
		xm=$m.$mode.${p}_lens_$j.$i
		cd $MODELS_PATH
		$ROOT/tools/pythia/model_xform.py --predict $p --mode $mode --num_layers $i --head $m.$mode.$p.$j $m $xm || break
		cd -
		./tools/w.sh $xm || break
		rm -rf $xm
	done
done


