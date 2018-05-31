NUM_ARTICLES=$(ls 230_tokenize -F |grep -v / | wc -l)
N_DEV=$('$NUM_ARTICLES' * 0.2)
N_TEST=$('$NUM_ARTICLES' * 0.1)
mkdir train
mkdir dev
mkdir tests

ls 230_tokenize/ |sort -R |tail -$N_DEV |while read file; do
    mv '$file' dev
done

ls 230_tokenize/ |sort -R |tail -$N_TEST |while read file; do
    mv '$file' tests
done

mv 230_tokenize/* train