#!/bin/sh
for i in L???.cif; do
    echo "Processing: $i"
    base="${i%.cif}"
    cif_modifier.py "$i" -mi w-2678 c-1 --clayff -o "${base}_w2750_c.cif"
#    cif_modifier.py "$i" -mi w-1303 --clayff -o "${base}_w1375.cif"
#    cif_modifier.py "$i" -mi w-1028 --clayff -o "${base}_w1100.cif"
#    cif_modifier.py "$i" -mi w-753 --clayff -o "${base}_w0825.cif"
done
