#!/bin/sh

steps=0
for i in *.py; do
    steps=$((steps + $(wc -l < "$i")))
done
echo "$steps"
