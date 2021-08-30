python -c "import nltk;nltk.download('wordnet')"

# Generate WNIDs
for dataset in CIFAR10 CIFAR100 Office_Home;
do
  nbdt-wnids --dataset=${dataset}
done;

# Generate and test hierarchies
for dataset in CIFAR10 CIFAR100 TinyImagenet200 Office_Home;
do
  nbdt-hierarchy --dataset=${dataset} --method=wordnet;
done;
