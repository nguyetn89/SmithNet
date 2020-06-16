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
mkdir Test_gt
mv Test/Test*_gt Test_gt/
cd ../UCSDped1
mkdir Test_gt
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

# Subway Entrance
mkdir Entrance
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1W3-KRkYyQW3vUD_LukZ2V3EAtUayz-TA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1W3-KRkYyQW3vUD_LukZ2V3EAtUayz-TA" -O ./Entrance/Entrance.avi && rm -rf /tmp/cookies.txt

# Subway Exit
mkdir Exit
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1A07Zg-5uvR5Iq9JtZX_CzkXyT6fvMQID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1A07Zg-5uvR5Iq9JtZX_CzkXyT6fvMQID" -O ./Exit/Exit.avi && rm -rf /tmp/cookies.txt  
