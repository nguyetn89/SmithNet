task=$1
dataset="UCSDped2"
workspace="./workspace_ICCV_extend"
if [ $task == "train" ]
then
    python ./main.py --dataset $dataset --height 128 --width 192 --task train --epoch 20 --batch 4 --workspace $workspace
else
    for epoch in $(seq 5 5 20)
    do
        python ./main.py --dataset $dataset --height 128 --width 192 --task infer --subset test --epoch $epoch --batch 4 --workspace $workspace
        python ./main.py --dataset $dataset --height 128 --width 192 --task infer --subset train --epoch $epoch --batch 4 --workspace $workspace
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
                    python ./main.py --dataset $dataset --height 128 --width 192 --task eval --epoch ${epoch} --workspace $workspace --patch ${patch} --stride ${stride} --power ${power} --print 0
                done
            done
        done
    done
fi
