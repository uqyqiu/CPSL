# !/bin/bash
 for seed in 32 42 52 62 72
 do
        python3 ../main_kmnist.py --batch_size=512 \
        --A_prop=0.01 \
        --im_ratio=0.1 \
        --sampling_num=1000 \
        --seed=$seed \
        --test_imb=True \
        --temperature=0.9 \
        --test_interval=50 \
        --g_epochs=10 \
        --c_epochs=100 \
        --linear_epochs=50 \
        --selection_method='kmeans_select' \
        --sup_epochs=20 \
        --reason4Exp="Experiments_NumSam:1000_20" \
        --hyperpara="fix_tau_:${seed}"
done