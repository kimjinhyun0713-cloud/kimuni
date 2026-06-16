#!/bin/sh


for name in "kim_crystal00" "kim_crystal01" "kim_tobermorite" "kim_bridgeCa00" "kim_bridgeCa01" "kim_bridgeCa02"; do
    echo -n $name
    kim.sh "${name}.cif" -mi w-582 --clayff -o "${name}.w50" -r --mode 3
done
	    
