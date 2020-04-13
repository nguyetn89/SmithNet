task=$1
skip="none"
dataset="UCSDped2"
workspace="./workspace_ICCV_extended"
if [ $task == "train" ]
then
    python ./main.py --dataset $dataset --height 128 --width 192 --RNN 1 --cat_latent 1 --elenorm 1 --chanorm 1 --sigmoid_instead_tanh 0 --training_gamma 0.9 --skip_blocks $skip --task train --epoch 20 --batch 16 --workspace $workspace --prt_summary 1
else
    for epoch in $(seq 5 5 20)
    do
        python ./main.py --dataset $dataset --height 128 --width 192 --RNN 1 --cat_latent 1 --elenorm 1 --chanorm 1 --sigmoid_instead_tanh 0 --training_gamma 0.9 --skip_blocks $skip --task infer --subset test --epoch $epoch --batch 16 --workspace $workspace --prt_summary 1
        python ./main.py --dataset $dataset --height 128 --width 192 --RNN 1 --cat_latent 1 --elenorm 1 --chanorm 1 --sigmoid_instead_tanh 0 --training_gamma 0.9 --skip_blocks $skip --task infer --subset train --epoch $epoch --batch 16 --workspace $workspace --prt_summary 1
    done
    for epoch in $(seq 5 5 20)
    do
        for power in $(seq 1 2)
        do
            for patch in $(seq 3 2 9)
            do
                for stride in $(seq 1 7)
                do
                    echo "epoch ${epoch}: power ${power}, patch ${patch}, stride ${stride}"
                    python ./main.py --dataset $dataset --height 128 --width 192 --RNN 1 --cat_latent 1 --elenorm 1 --chanorm 1 --sigmoid_instead_tanh 0 --training_gamma 0.9 --skip_blocks $skip --task eval --epoch ${epoch} --workspace $workspace --patch ${patch} --stride ${stride} --power ${power} --print 0
                done
            done
        done
    done
fi
