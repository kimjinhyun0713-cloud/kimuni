cif_head  = """
#======================================================================
# CRYSTAL DATA
#----------------------------------------------------------------------
data_VESTA_phase_1

_chemical_name_common                  '{}'
_cell_length_a                          {:20.16f}
_cell_length_b                          {:20.16f}
_cell_length_c                          {:20.16f}
_cell_angle_alpha                       {:20.16f}
_cell_angle_beta                        {:20.16f}
_cell_angle_gamma                       {:20.16f}
_space_group_name_H-M_alt              'P 1'
_space_group_IT_number                 1

loop_
_space_group_symop_operation_xyz
   'x, y, z'

loop_
   _atom_site_label
   _atom_site_occupancy
   _atom_site_fract_x
   _atom_site_fract_y
   _atom_site_fract_z
   _atom_site_adp_type
   _atom_site_U_iso_or_equiv
   _atom_site_type_symbol
""".strip()

cif_tail = "{:>5} {:>10.7f} {:>15.10f} {:>15.10f} {:>15.10f} {:>5s} {:>8.7f} {:>5s}"


lmp_head_unwrapped = """
ITEM: TIMESTEP
{}
ITEM: NUMBER OF ATOMS
{}
ITEM: BOX BOUNDS xy xz yz pp pp pp
{}
ITEM: ATOMS id type element xu yu zu mol
""".strip()


lmp_head = """
ITEM: TIMESTEP
{}
ITEM: NUMBER OF ATOMS
{}
ITEM: BOX BOUNDS xy xz yz pp pp pp
{}
ITEM: ATOMS id type element xu yu zu mol
""".strip()


SCAN_D3 = """
IVDW   = 12
VDW_S8 = 0 
VDW_A1 = 0.538
VDW_A2 = 5.4200
""".strip()
