variable        T string 330
log             ${base}.log
units           metal
atom_style      atomic
read_data       ../../${base}.data
velocity        all create ${T} ${seed}
pair_style      deepmd ../../hoge.pb
pair_coeff      * * ${elem}
timestep        0.0001
thermo          1000
# minimize        1.0e-4 1.0e-6 100 1000
				
group           layer type 1 6 8 9 			
thermo_style    custom step etotal temp press density
dump            1 all custom 1000 ${base}.lammpstrj id type element x y z
dump_modify     1 element ${elem} sort id
fix             2 layer recenter INIT INIT INIT 

fix             1 all nvt temp $T $T $(dt*10)
run             50000
unfix           1
fix             2 layer recenter INIT INIT INIT 
fix             1 all npt temp $T $T $(dt*100) tri 1.01325 1.01325 $(dt*1000)
run             50000
write_restart   ${base}.rst
write_data      ${base}.data
