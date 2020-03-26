task=$1
dataset="Avenue"
workspace="./workspace_flow_unetInstant_cross"
unet="instant"
if [ $task == "train" ]
then
    python ./main.py --dataset $dataset --height 128 --width 192 --task train --epoch 20 --batch 4 --workspace $workspace --unet $unet --cross_pred 1
else
    for epoch in $(seq 5 5 25)
    do
        python ./main.py --dataset $dataset --height 128 --width 192 --task infer --epoch $epoch --batch 4 --workspace $workspace --unet $unet --cross_pred 1
    done
    for epoch in $(seq 5 5 25)
    do
        for power in $(seq 1 2)
        do
            for patch in $(seq 3 2 9)
            do
                for stride in $(seq 1 7)
                do
                    echo "epoch ${epoch}: power ${power}, patch ${patch}, stride ${stride}"
                    python ./main.py --dataset $dataset --height 128 --width 192 --task eval --epoch ${epoch} --batch 4 --workspace $workspace --unet $unet --cross_pred 1 --print 0
                done
            done
        done
    done
fi
