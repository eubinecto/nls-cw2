echo "##### BASE #####"
python3 -m  nls_cw2.task_2.part_b.evaluate_models \
  --model="base"

echo "##### BOW #####"
echo "----with no additional features ---"
python3 -m  nls_cw2.task_2.part_b.evaluate_models \
  --model="bow" --k=4

echo "---with additional features ---"
python3 -m  nls_cw2.task_2.part_b.evaluate_models \
  --model="bow" --k=4 --more_features

echo "##### W2V #####"
echo "----with no additional features ---"
python3 -m  nls_cw2.task_2.part_b.evaluate_models \
  --model="w2v" --k=4

echo "---with additional features ---"
python3 -m  nls_cw2.task_2.part_b.evaluate_models \
  --model="w2v" --k=4 --mor