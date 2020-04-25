# Avenue
wget http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip
unzip -q Avenue_Dataset.zip
rm Avenue_Dataset.zip
mv "./Avenue Dataset" "./Avenue_Dataset"
mkdir "./Avenue_Dataset/testing_gt"
#cd "./Avenue_Dataset/testing_gt"
#unzip Avenue_testing_label_mask.zip
#cd ../../

# UCSD Peds
wget http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz
tar -xf UCSD_Anomaly_Dataset.tar.gz
rm UCSD_Anomaly_Dataset.tar.gz
cd ./UCSD_Anomaly_Dataset.v1p2/UCSDped2
mv Test/Test*_gt Test_gt/
cd ../../

# Traffic-Belleview
wget http://vision.eecs.yorku.ca/research/anomalous-behaviour-data/sets/Traffic-Belleview.zip -O Belleview.zip
unzip -q Belleview.zip
rm Belleview.zip

# Traffic-Train
wget http://vision.eecs.yorku.ca/research/anomalous-behaviour-data/sets/Traffic-Train.zip -O Train.zip
unzip -q Train.zip
rm Train.zip

# ShanghaiTech
wget --no-check-certificate "https://onedrive.live.com/download?cid=3705E349C336415F&resid=3705E349C336415F%2172436&authkey=%21AMqh2fTSemfrokE" -O ShanghaiTech.tar.gz
tar -xf ShanghaiTech.tar.gz
rm ShanghaiTech.tar.gz
