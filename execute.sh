# python ./main.py --dataset UCSDped2 --height 128 --width 192 --task train --epoch 20 --batch 4
for epoch in $(seq 5 5 20)
do
    for power in $(seq 1 2)
    do
        for patch in $(seq 3 2 9)
        do
            for stride in $(seq 1 7)
            do
                echo "epoch ${epoch}: power ${power}, patch ${patch}, stride ${stride}"
                python ./main.py --dataset UCSDped2 --height 128 --width 192 --task eval --epoch ${epoch} --batch 4
            done
        done
    done
done
