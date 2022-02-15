if [ ! -d "./data" ]
then 
    echo "Downloading Dataset"

    mkdir data/
fi

cd data/

if [ ! -d "./handwrittendataset/" ]
then
    curl https://archive.ics.uci.edu/ml/machine-learning-databases/00389/DevanagariHandwrittenCharacterDataset.zip --output handwrittendataset.zip
    mv handwrittendataset.zip data/handwrittendataset.zip

    cd data/

    unzip handwrittendataset.zip


    rm handwrittendataset.zip

    mv -r DevanagariHandwrittenCharacterDataset/ handwrittendataset/

    cd ../..

else 
    echo "Dataset found"

fi