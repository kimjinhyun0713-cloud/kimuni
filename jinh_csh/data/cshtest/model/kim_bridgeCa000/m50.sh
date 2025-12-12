#!/bin/sh


for name in "kim_bridgeCa000"; do
    echo -n $name
    kim.sh "${name}.cif" -mi w-582 --clayff -o "${name}.w50" -r --mode 3
done
	    
