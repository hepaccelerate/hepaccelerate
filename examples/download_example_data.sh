#!/bin/bash

if [ "$#" -ne 1 ]
then
  echo "Usage: ./examples/download_example_data.sh /path/to/output/folder"
  exit 1
fi

TARGET_DIR=$1
mkdir -p $TARGET_DIR
echo "Going to download 144GB of data to $TARGET_DIR"

for fn in `cat examples/data.txt`; do
    dir=`dirname $fn`
    if [ ! -d "$TARGET_DIR/$dir" ]; then
        mkdir -p "$TARGET_DIR/$dir"
    fi
    curl -k https://jpata.web.cern.ch/jpata/$fn -o "$TARGET_DIR/$fn"
done
