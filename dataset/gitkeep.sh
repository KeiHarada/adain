#!/bin/bash
dir_path="./**/*"
dir=`find $dir_path -type d`
for dir_i in $dir
do
    echo $dir_i"/.gitkeep"
    touch $dir_i"/.gitkeep"
done