METHOD=Default
echo "==================================================== ${METHOD} ===================================================="
for DATASET in SICK SICK-adjacent STSb STSb-adjacent
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

METHOD=SAGPooling
echo "==================================================== ${METHOD} ===================================================="

for DATASET in SICK SICK-adjacent STSb STSb-adjacent
do
 echo "---------------------------------------------------- Started ${DATASET} Training ----------------------------------------------------"
 echo ""
 echo ""
 for NUM in 1 2 3 4 5
 do
 echo "_____________________________ Training Count ${NUM}  _____________________________"
 echo ""
 python ./calc_similarity/src/main.py --use-sagpool --train-dir ./dataset/w2v/${DATASET}/train --test-dir ./dataset/w2v/${DATASET}/test --valid-dir ./dataset/w2v/${DATASET}/validation --early-stop 20 -d 0.7 -e 500 --save-path ./models/${DATASET}/${METHOD}/${NUM} --tensorboard-dir ./TensorBoard/${DATASET}/${METHOD}/${NUM}
 done
 echo "---------------------------------------------------- Ended ${DATASET} Training ----------------------------------------------------"
 echo ""
 echo ""
done

METHOD=GraphSAGE
echo "==================================================== ${METHOD} ===================================================="

for DATASET in SICK SICK-adjacent STSb STSb-adjacent
do
 echo "---------------------------------------------------- Started ${DATASET} Training ----------------------------------------------------"
 echo ""
 echo ""
 for NUM in 1 2 3 4 5
 do
 echo "_____________________________ Training Count ${NUM}  _____________________________"
 echo ""
 python ./calc_similarity/src/main.py --use-sage --train-dir ./dataset/w2v/${DATASET}/train --test-dir ./dataset/w2v/${DATASET}/test --valid-dir ./dataset/w2v/${DATASET}/validation --early-stop 20 -d 0.7 -e 500 --save-path ./models/${DATASET}/${METHOD}/${NUM} --tensorboard-dir ./TensorBoard/${DATASET}/${METHOD}/${NUM}
 done
 echo "---------------------------------------------------- Ended ${DATASET} Training ----------------------------------------------------"
 echo ""
 echo ""
done

METHOD=SAGEPooling
echo "==================================================== ${METHOD} ===================================================="

for DATASET in SICK SICK-adjacent STSb STSb-adjacent
do
 echo "---------------------------------------------------- Started ${DATASET} Training ----------------------------------------------------"
 echo ""
 echo ""
 for NUM in 1 2 3 4 5
 do
 echo "_____________________________ Training Count ${NUM}  _____________________________"
 echo ""
 python ./calc_similarity/src/main.py --use-sage --use-sagpool --train-dir ./dataset/w2v/${DATASET}/train --test-dir ./dataset/w2v/${DATASET}/test --valid-dir ./dataset/w2v/${DATASET}/validation --early-stop 20 -d 0.7 -e 500 --save-path ./models/${DATASET}/${METHOD}/${NUM} --tensorboard-dir ./TensorBoard/${DATASET}/${METHOD}/${NUM}
 done
 echo "---------------------------------------------------- Ended ${DATASET} Training ----------------------------------------------------"
 echo ""
 echo ""
done

METHOD=cos
echo "==================================================== ${METHOD} ===================================================="
for DATASET in SICK SICK-adjacent STSb STSb-adjacent
do
 echo "---------------------------------------------------- Started ${DATASET} Training ----------------------------------------------------"
 echo ""
 echo ""
 for NUM in 1 2 3 4 5
 do
 echo "_____________________________ Training Count ${NUM}  _____________________________"
 echo ""
 python ./calc_similarity/src/main.py --use-cos --train-dir ./dataset/w2v/${DATASET}/train --test-dir ./dataset/w2v/${DATASET}/test --valid-dir ./dataset/w2v/${DATASET}/validation --early-stop 20 -d 0.7 -e 500 --save-path ./models/${DATASET}/${METHOD}/${NUM} --tensorboard-dir ./TensorBoard/${DATASET}/${METHOD}/${NUM}
 done
 echo "---------------------------------------------------- Ended ${DATASET} Training ----------------------------------------------------"
 echo ""
 echo ""
done

METHOD=SAGPool-cos
echo "==================================================== ${METHOD} ===================================================="

for DATASET in SICK SICK-adjacent STSb STSb-adjacent
do
 echo "---------------------------------------------------- Started ${DATASET} Training ----------------------------------------------------"
 echo ""
 echo ""
 for NUM in 1 2 3 4 5
 do
 echo "_____________________________ Training Count ${NUM}  _____________________________"
 echo ""
 python ./calc_similarity/src/main.py --use-cos --use-sagpool --train-dir ./dataset/w2v/${DATASET}/train --test-dir ./dataset/w2v/${DATASET}/test --valid-dir ./dataset/w2v/${DATASET}/validation --early-stop 20 -d 0.7 -e 500 --save-path ./models/${DATASET}/${METHOD}/${NUM} --tensorboard-dir ./TensorBoard/${DATASET}/${METHOD}/${NUM}
 done
 echo "---------------------------------------------------- Ended ${DATASET} Training ----------------------------------------------------"
 echo ""
 echo ""
done

METHOD=GraphSAGE-cos
echo "==================================================== ${METHOD} ===================================================="

for DATASET in SICK SICK-adjacent STSb STSb-adjacent
do
 echo "---------------------------------------------------- Started ${DATASET} Training ----------------------------------------------------"
 echo ""
 echo ""
 for NUM in 1 2 3 4 5
 do
 echo "_____________________________ Training Count ${NUM}  _____________________________"
 echo ""
 python ./calc_similarity/src/main.py --use-cos --use-sage --train-dir ./dataset/w2v/${DATASET}/train --test-dir ./dataset/w2v/${DATASET}/test --valid-dir ./dataset/w2v/${DATASET}/validation --early-stop 20 -d 0.7 -e 500 --save-path ./models/${DATASET}/${METHOD}/${NUM} --tensorboard-dir ./TensorBoard/${DATASET}/${METHOD}/${NUM}
 done
 echo "---------------------------------------------------- Ended ${DATASET} Training ----------------------------------------------------"
 echo ""
 echo ""
done

METHOD=SAGESAG-cos
echo "==================================================== ${METHOD} ===================================================="

for DATASET in SICK SICK-adjacent STSb STSb-adjacent
do
 echo "---------------------------------------------------- Started ${DATASET} Training ----------------------------------------------------"
 echo ""
 echo ""
 for NUM in 1 2 3 4 5
 do
 echo "_____________________________ Training Count ${NUM}  _____________________________"
 echo ""
 python ./calc_similarity/src/main.py --use-cos --use-sage --use-sagpool --train-dir ./dataset/w2v/${DATASET}/train --test-dir ./dataset/w2v/${DATASET}/test --valid-dir ./dataset/w2v/${DATASET}/validation --early-stop 20 -d 0.7 -e 500 --save-path ./models/${DATASET}/${METHOD}/${NUM} --tensorboard-dir ./TensorBoard/${DATASET}/${METHOD}/${NUM}
 done
 echo "---------------------------------------------------- Ended ${DATASET} Training ----------------------------------------------------"
 echo ""
 echo ""
done