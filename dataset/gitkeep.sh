#!/bin/bash

dir_path="./*"
dirs=`find $dir_path -type d`

for dir in $dirs;
do
    echo $dir
    touch $dir"/.gitkeep"
done