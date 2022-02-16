mode=extra_linear
p=self
MODELS_PATH=/mnt/ssd-1/igor/gpt-neox/models
ROOT=~/igor/gpt-neox

for m in dense_medium dense_large; do
	for j in {0..24}; do
		hm=$m.$mode.$p.$j
		./tools/w.sh $hm || break

		for i in {0..24}; do
			xm=$m.$mode.${p}_lens_$j.$i
			cd $MODELS_PATH
			$ROOT/tools/pythia/model_xform.py --predict $p --mode $mode --num_layers $i --head $hm $m $xm || break
			cd -
			./tools/w.sh $xm || break 2
			rm -rf $MODELS_PATH/$xm
		done
	done
done
