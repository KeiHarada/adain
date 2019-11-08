#!/bin/bash

for i in 1 2 3
do
    mkdir "./train_TianJin"$i
    touch "./train_TianJin"$i"/.gitkeep"
done

for dir_i in BeiJing ShenZhen TianJin XiangGang ShiJiaZhuang GuangZhou TangShan QinHuangDao BaoDing ZhangJiaKou ChengDe CangZhou LangFang HengShui DongGuan FoShan HuiZhou JiangMen ShanTou ZiBo
do
    echo $dir_i
    for loop in 1 2 3
    do
        mkdir "./test_"$dir_i"$loop"
        touch "./test_"$dir_i$loop"/.gitkeep"
    done
done