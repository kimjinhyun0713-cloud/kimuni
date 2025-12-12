#!/bin/sh

for name in "kim_bridgeCa000"; do
    echo -n $name
    kim.sh "${name}.cif" -mi w-278 --clayff -o "${name}.w40" -r 
done
	    
