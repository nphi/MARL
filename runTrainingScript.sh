#!/bin/bash

# Set variables
folderName="$1"
dirOut="/home/ubuntu/Nguyen/Data/"
fullPath="$dirOut$folderName"
trainOut="$2"
arena="$3"
clip="$4"
cpu="$5"
rec="$6"
inp="$7"
iter="$8"
model="$9"

# Create directory and start screen session
echo "$fullPath"
if [ ! -d "$fullPath" ]; then
  mkdir "$fullPath"
  mkdir "$fullPath/out_files"
  mkdir "$fullPath/raster"
  mkdir "$fullPath/training_checkpoints/"
  mkdir "$fullPath/training_result/"
  mkdir "$fullPath/analysis_result/" 
else
 echo "Directory already exists: $fullPath"
fi

screen -dmS "$folderName" bash -c "python /home/ubuntu/Nguyen/Code/NP/python/MultiAgent_Train_v2.py \
  --dir-out "$dirOut$trainOut/training_checkpoints" \
  --arena-file "$arena" \
  --clip-param "$clip" \
  --num-workers "$cpu" \
  --l2-curr "$rec" \
  --train-iter "$iter" \
  --l2-inp "$inp" \
  --model-file "$model"> \"$fullPath/training_checkpoints/log.txt\" 2>&1"
