cif_head  = """
#======================================================================
# CRYSTAL DATA
#----------------------------------------------------------------------
data_VESTA_phase_1

_chemical_name_common                  '{}'
_cell_length_a                         {}
_cell_length_b                         {}
_cell_length_c                         {}
_cell_angle_alpha                      {}
_cell_angle_beta                       {}
_cell_angle_gamma                      {}
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
"""

cif_tail = "{:>5} {:>10.7f} {:>15.10f} {:>15.10f} {:>15.10f} {:>5s} {:>8.7f} {:>5s}"
