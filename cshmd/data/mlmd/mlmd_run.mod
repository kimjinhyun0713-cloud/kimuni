variable        startstep equal "step*0.001"
variable        fstep format startstep %05.0f
#log             ${base}.*.log
read_restart    ${base}.rst
pair_style      deepmd ../../hoge.pb
pair_coeff      * * ${elem}
thermo_style    custom step etotal temp press vol density
thermo          1000
timestep        0.0005
group           layer type 1 6 8 9
log             ${base}.${fstep}.log
reset_timestep  0		      
dump            2 all custom 1000 ${base}.${fstep}.lammpstrj id type element x y z
dump_modify     2 element ${elem} sort id
fix             1 all npt temp 330 330 $(dt * 100) aniso 1.01325 1.01325 $(dt * 1000)
fix             2 layer recenter INIT INIT INIT
variable        i loop 4
label           loop
run             2000
write_restart   rst.*
write_data      ${base}.data.*
next i
jump            SELF loop
