METHOD=Default
echo "==================================================== Default ===================================================="
for DATASET in STSb STSb-adjacent
do
 echo "---------------------------------------------------- Started ${DATASET} Training ----------------------------------------------------"
 echo ""
 echo ""
 for NUM in 1 2 3 4 5
 do
 echo "_____________________________ Training Count ${NUM}  _____________________________"
 echo ""
 python ./calc_similarity/src/main.py --train-dir ./dataset/w2v/${DATASET}/train --test-dir ./dataset/w2v/${DATASET}/test --valid-dir ./dataset/w2v/${DATASET}/validation --early-stop 20 -d 0.7 -e 500 --save-path ./models/${DATASET}/${METHOD}/${NUM} --tensorboard-dir ./TensorBoard/${DATASET}/${METHOD}/${NUM}
 done
 echo "---------------------------------------------------- Ended ${DATASET} Training ----------------------------------------------------"
 echo ""
 echo ""
done

# METHOD=SAGEPoolimg
# echo "==================================================== GraphSAGE SAGPooling ===================================================="

# for DATASET in SICK SICK-adjacent STSb STSb-adjacent
# do
#  echo "---------------------------------------------------- Started ${DATASET} Training ----------------------------------------------------"
#  echo ""
#  echo ""
#  for NUM in 1 2 3 4 5
#  do
#  echo "_____________________________ Training Count ${NUM}  _____________________________"
#  echo ""
#  python ./calc_similarity/src/main.py --use-sage --use-sagpool --train-dir ./dataset/w2v/${DATASET}/train --test-dir ./dataset/w2v/${DATASET}/test --valid-dir ./dataset/w2v/${DATASET}/validation --early-stop 20 -d 0.7 -e 500 --save-path ./models/${DATASET}/${METHOD}/${NUM} --tensorboard-dir ./TensorBoard/${DATASET}/${METHOD}/${NUM}
#  done
#  echo "---------------------------------------------------- Ended ${DATASET} Training ----------------------------------------------------"
#  echo ""
#  echo ""
# done

python ./calc_similarity/src/main.py --train-dir ./dataset/w2v/STSb/train --test-dir ./dataset/w2v/STSb/test --valid-dir ./dataset/w2v/STSb/validation --early-stop 20 -d 0.7 -e 500 --save-path ./models/STSb/Default/fk --tensorboard-dir ./TensorBoard/STSb/Default/fk