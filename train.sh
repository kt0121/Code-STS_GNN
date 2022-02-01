DATASET1=SICK-adjacent
DATASET2=STSb-adjacent
for NUM in 1 2 3 4 5
do
 echo "_____________________________ Training Count ${NUM}  _____________________________"
 echo ""
 python ./calc_similarity/src/main.py \
 --train-dir ./dataset/${DATASET1}/train ./dataset/${DATASET2}/train \
 --test-dir ./dataset/${DATASET1}/test ./dataset/${DATASET2}/test \
 --valid-dir ./dataset/${DATASET1}/validation ./dataset/${DATASET2}/validation \
 --early-stop 50 -d 0.7 -e 500 --save-path ./models/BIG/${NUM} \
 --tensorboard-dir ./TensorBoard/BIG/${NUM}
done