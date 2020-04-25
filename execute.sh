dataset=$1
task=$2
skip="none"
workspace="./workspace_ICCV_extended_fixed_RNN"
RNN=1
cat_latent=1
elenorm=1
chanorm=1
sigmoid=1
gamma=-1
echo "${dataset}: RNN ${RNN} cat_latent ${cat_latent} elenorm ${elenorm} chanorm ${chanorm} sigmoid_instead_tanh ${sigmoid} training_gamma ${gamma}"
if [ $task == "train" ]
then
    python ./main.py --dataset $dataset --height 128 --width 192 --RNN $RNN --cat_latent $cat_latent --elenorm $elenorm --chanorm $chanorm --sigmoid_instead_tanh $sigmoid --training_gamma $gamma --skip_blocks $skip --task train --epoch 30 --batch 16 --workspace $workspace --prt_summary 0
else
    for epoch in $(seq 5 5 30)
    do
        python ./main.py --dataset $dataset --height 128 --width 192 --RNN $RNN --cat_latent $cat_latent --elenorm $elenorm --chanorm $chanorm --sigmoid_instead_tanh $sigmoid --training_gamma $gamma --skip_blocks $skip --task infer --subset test --epoch $epoch --batch 16 --workspace $workspace --prt_summary 0
        python ./main.py --dataset $dataset --height 128 --width 192 --RNN $RNN --cat_latent $cat_latent --elenorm $elenorm --chanorm $chanorm --sigmoid_instead_tanh $sigmoid --training_gamma $gamma --skip_blocks $skip --task infer --subset train --epoch $epoch --batch 16 --workspace $workspace --prt_summary 0
    done
    for epoch in $(seq 5 5 30)
    do
        for power in $(seq 1 2)
        do
            for patch in $(seq 3 2 9)
            do
                for stride in $(seq 1 7)
                do
                    echo "epoch ${epoch}: power ${power}, patch ${patch}, stride ${stride}"
                    python ./main.py --dataset $dataset --height 128 --width 192 --RNN $RNN --cat_latent $cat_latent --elenorm $elenorm --chanorm $chanorm --sigmoid_instead_tanh $sigmoid --training_gamma $gamma --skip_blocks $skip --task eval --epoch ${epoch} --workspace $workspace --patch ${patch} --stride ${stride} --power ${power} --print 0
                done
            done
        done
    done
fi
