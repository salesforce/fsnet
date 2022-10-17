## M

i=1
ns=(1 )
bszs=(1 )
lens=(1 24 48)
methods=('ogd' 'large' 'er' 'derpp' 'fsnet' 'nomem' 'naive')
for n in ${ns[*]}; do
for bsz in ${bszs[*]}; do
for len in ${lens[*]}; do
for m in ${methods[*]}; do
CUDA_VISIBLE_DEVICES=0 python -u main.py --method $m --root_path ./data/ --n_inner $n --test_bsz $bsz --data ETTh2 --features M --seq_len 60 --label_len 0 --pred_len $len --des 'Exp' --itr $i --train_epochs 6 --learning_rate 1e-3 --online_learning 'full'
CUDA_VISIBLE_DEVICES=0 python -u main.py --method $m --root_path ./data/ --n_inner $n --test_bsz $bsz --data ETTm1 --features M --seq_len 60 --label_len 0 --pred_len $len --des 'Exp' --itr $i --train_epochs 6 --learning_rate 1e-3 --online_learning 'full'
CUDA_VISIBLE_DEVICES=0 python -u main.py  --method $m --root_path ./data/ --n_inner $n --test_bsz $bsz --data WTH --features M --seq_len 60 --label_len 0 --pred_len $len --des 'Exp' --itr $i --train_epochs 6 --learning_rate 1e-3 --online_learning 'full'
CUDA_VISIBLE_DEVICES=0 python -u main.py --method $m --root_path ./data/ --n_inner $n --test_bsz $bsz --data ECL --features M --seq_len 60 --label_len 0 --pred_len $len --des 'Exp' --itr $i --train_epochs 6 --learning_rate 3e-3 --online_learning 'full'
done
done
done
done
# M
lens=(24 1)
for n in ${ns[*]}; do
for bsz in ${bszs[*]}; do
for len in ${lens[*]}; do
for m in ${methods[*]}; do
echo $m $l
CUDA_VISIBLE_DEVICES=0 python -u main.py --method $m --root_path ./data/ --n_inner $n --test_bsz $bsz --data Traffic --features M --seq_len 60 --label_len 0 --pred_len $len --des 'Exp' --itr $i --train_epochs 6 --learning_rate 3e-3 --online_learning 'full'
done
done
done
done










