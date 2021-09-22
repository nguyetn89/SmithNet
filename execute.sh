dataset=$1
task=$2
skip="none"
workspace="./workspace_ICCV_extended_fixed_RNN"
RNN=1
cat_latent=1
elenorm=1
sigmoid=1
gamma=0.9
chanorm=1
relu_chanorm=1
echo "${dataset}: RNN ${RNN} cat_latent ${cat_latent} elenorm ${elenorm} sigmoid_instead_tanh ${sigmoid} training_gamma ${gamma} chanorm ${chanorm} relu_chanorm ${relu_chanorm} skip_blocks ${skip}"
epoch=60
if [ $task == "train" ]
then
    python ./main.py --dataset $dataset --height 128 --width 192 \
        --RNN $RNN --cat_latent $cat_latent --elenorm $elenorm --sigmoid_instead_tanh $sigmoid \
        --training_gamma $gamma --chanorm $chanorm --relu_chanorm $relu_chanorm \
        --skip_blocks $skip --task train --epoch $epoch --batch 16 \
        --workspace $workspace --prt_summary 0 --const_lambda 0.2
else
    python ./main.py --dataset $dataset --height 128 --width 192 \
        --RNN $RNN --cat_latent $cat_latent --elenorm $elenorm --sigmoid_instead_tanh $sigmoid \
        --training_gamma $gamma --chanorm $chanorm --relu_chanorm $relu_chanorm \
        --skip_blocks $skip --task infer --subset test --epoch $epoch --batch 16 \
        --workspace $workspace --prt_summary 0 --const_lambda 0.2

    python ./main.py --dataset $dataset --height 128 --width 192 \
        --RNN $RNN --cat_latent $cat_latent --elenorm $elenorm --sigmoid_instead_tanh $sigmoid \
        --training_gamma $gamma --chanorm $chanorm --relu_chanorm $relu_chanorm \
        --skip_blocks $skip --task infer --subset train --epoch $epoch --batch 16 \
        --workspace $workspace --prt_summary 0 --const_lambda 0.2

    power=1
    patch=7
    stride=6
    for lambdaval in $(seq 0.2 0.2 1)
    do
        echo "epoch ${epoch}: power ${power}, patch ${patch}, stride ${stride}, lambda ${lambdaval}"
        python ./main.py --dataset $dataset --height 128 --width 192 \
                --RNN $RNN --cat_latent $cat_latent \
                --elenorm $elenorm --sigmoid_instead_tanh $sigmoid --training_gamma $gamma \
                --chanorm $chanorm --relu_chanorm $relu_chanorm \
                --skip_blocks $skip --task eval --epoch ${epoch} \
                --workspace $workspace --patch ${patch} --stride ${stride} --power ${power} \
                --print 0 --const_lambda ${lambdaval}
    done
fi
