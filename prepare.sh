# 解压数据集中附带的官方处理代码
# unzip -nq -d ./ dataset/cityscapesscripts.zip 

# # 解压数据集中的gtFine
unzip -nq -d dataset/gtFine/ dataset/gtFine_train.zip
unzip -nq -d dataset/gtFine/ dataset/gtFine_val.zip
unzip -nq -d dataset/gtFine/ dataset/gtFine_test.zip
# # 解压数据集中的leftImg8bit
unzip -nq -d dataset/leftImg8bit/ dataset/leftImg8bit_train.zip
unzip -nq -d dataset/leftImg8bit/ dataset/leftImg8bit_val.zip
unzip -nq -d dataset/leftImg8bit/ dataset/leftImg8bit_test.zip
# python cityscapesscripts/preparation/createTrainIdLabelImgs.py