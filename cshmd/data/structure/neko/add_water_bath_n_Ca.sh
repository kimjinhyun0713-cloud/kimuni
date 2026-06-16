#!/bin/sh
for i in L*.cif; do
    echo -n $i;
    cif_modifier.py $i -e 0 0 1.5  -o $i -r;
    cif_modifier.py $i -mi w-1310 Ca-10 --clayff -o "Ca10w$i";
done
