#!/bin/sh
for i in  LiH2.cif  L5H2.cif L5H0H2.cif  L2H0H2.cif L2H2.cif; do
    echo -n $i;
    cif_modifier.py $i -e 0 0 1.5  -o $i -r;
    cif_modifier.py $i -mi w-1310 --clayff -o "w$i";
done
