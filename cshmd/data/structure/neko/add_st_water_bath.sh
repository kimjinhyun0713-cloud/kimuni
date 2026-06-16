#!/bin/sh
for i in st*.cif; do
    echo -n $i;
    cif_modifier.py $i -mi w-1310 --clayff -o "w$i";
done
