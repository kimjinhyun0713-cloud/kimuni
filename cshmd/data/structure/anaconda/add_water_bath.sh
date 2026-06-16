#!/bin/sh
for i in *.cif; do
    echo "Processing: $i"

    base="${i%.cif}"

    cif_modifier.py "$i" -mi w-4878 --clayff -o "${base}_w4950.cif"
#    cif_modifier.py "$i" -mi w-1303 --clayff -o "${base}_w1375.cif"
#    cif_modifier.py "$i" -mi w-1028 --clayff -o "${base}_w1100.cif"
#    cif_modifier.py "$i" -mi w-753 --clayff -o "${base}_w0825.cif"
done
