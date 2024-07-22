import math
import numpy as np

# Import Pyomo libraries
from pyomo.environ import (
    Set,
    Var,
    check_optimal_termination,
    Param,
    Suffix,
    NonNegativeReals,
    value,
    log,
    Constraint,
    units as pyunits,
    sqrt,
)
from pyomo.common.config import Bool, ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
    UnitModelBlockData,
    useDefault,
)
from idaes.core.util.misc import add_object_reference
from watertap.core.solvers import get_solver
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.config import is_physical_parameter_block

from idaes.core.util.exceptions import ConfigurationError, InitializationError

import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core.util.constants import Constants
from enum import Enum

catalyst_correction = 10

kw = 10**-8 * pyunits.mol**2 * pyunits.meter**-6
kd_zero = 2 * 10**-6 * pyunits.second**-1
kr = 1 * 10**10 * pyunits.L * pyunits.mole**-1 * pyunits.second**-1
conc_water = 50 * pyunits.mol * pyunits.L**-1
# current_dens_lim_bpem = 1e2 * pyunits.amp * pyunits.meter ** -2
membrane_fixed_charge = 5e3 * pyunits.mol * pyunits.meter**-3
relative_permittivity = 30
temperature = 298.15 * pyunits.K

conc_salt = 1 * 2 * pyunits.mol * pyunits.L**-1
slit_length = 800e-6 * pyunits.meter
diffusion = (2.03 + 1.96) / 2 * 10**-9 * pyunits.meter**2 * pyunits.second**-1

i_lim_estimate = (
    diffusion
    * Constants.faraday_constant
    * conc_salt**2
    / membrane_fixed_charge
    / slit_length
)
i_lim_estimate = pyunits.convert(
    i_lim_estimate, to_units=pyunits.amp * pyunits.meter**-2
)


kr_loc = pyunits.convert(
    kr, to_units=pyunits.meter**3 * pyunits.mole**-1 * pyunits.second**-1
)
conc_water_loc = pyunits.convert(conc_water, to_units=pyunits.mol * pyunits.meter**-3)
frac = 1 * 10**-1
const = 0.0936 * pyunits.K**2 * pyunits.volt**-1 * pyunits.meter


qty_une = value(
    (
        (
            Constants.vacuum_electric_permittivity
            * relative_permittivity**2
            * temperature**2
            * Constants.avogadro_number
            * Constants.elemental_charge
        )
    )
    / value(const * Constants.faraday_constant * membrane_fixed_charge)
)

qty_deux = value(kd_zero * conc_water_loc)
qty_trois = value(kr_loc * kw)
y2 = value(i_lim_estimate * frac) / catalyst_correction
print("qty_deux = ", qty_deux)
print("qty_trois = ", qty_trois)
print("qty_une = ", qty_une)

qty_une_dim = (
    (
        Constants.vacuum_electric_permittivity
        * relative_permittivity**2
        * temperature**2
        * Constants.avogadro_number
        * Constants.elemental_charge
    )
) / (const * Constants.faraday_constant * membrane_fixed_charge)
qty_deux_dim = kd_zero * conc_water_loc
qty_trois_dim = kr_loc * kw
y2_dim = i_lim_estimate * frac


# dat =[y2, qty_une, qty_deux, qty_trois]
# print(dat)
#
# print("check: current_dens_lim_bpem = ", value(self.current_dens_lim_bpem[t]))
# self.eq_current_dens_lim_bpem()
# print("updated: current_dens_lim_bpem = ", value(self.current_dens_lim_bpem[t]))

terms = 40
gE_raw = []
for indx in range(terms):
    rev_indx = terms - indx - 1
    gE_raw.append(
        2**rev_indx / (math.factorial(rev_indx) * math.factorial(rev_indx + 1))
    )

gE = np.array(gE_raw)
print("gE2 post = ", gE)
gE2 = gE * qty_deux
print("gE2 pre = ", gE2)
gE2[-1] += -qty_trois
print("gE2 post = ", gE2)
qty = gE2
gE2_list = gE2.tolist()
coeff = qty.tolist() + [-y2 / qty_une]
print("qty = ", qty)
print("y2", y2, "...qty une = ", qty_une)
print("coeff = ", coeff)
x2 = np.roots(coeff)
x2_real = [root.real for root in x2 if np.isclose(root.imag, 0) and root.real > 0]
non_dim_field = np.array(x2_real)
field_generated = non_dim_field * relative_permittivity * temperature**2 / const
lambda_depletion = (
    field_generated
    * Constants.vacuum_electric_permittivity
    * relative_permittivity
    / (Constants.faraday_constant * membrane_fixed_charge)
)

print(f"Field enhancement = {np.polyval(gE_raw,x2_real)}")
print(f"production rate = {np.polyval(gE,x2_real)*qty_deux - qty_trois}")
print(
    f"current generated to limiting current= {(np.polyval(gE,x2_real)*qty_deux - qty_trois) * qty_une * x2_real/y2}"
)

var_check = qty_une_dim * (qty_deux_dim - qty_trois_dim) * non_dim_field / y2_dim

print("var_check = ", value(var_check), pyunits.get_units(var_check))
var_check_deux = (
    qty_une_dim
    * (qty_deux_dim * np.polyval(gE, x2_real) - qty_trois_dim)
    * non_dim_field
    / y2_dim
)

print("var_check = ", value(var_check_deux), pyunits.get_units(var_check_deux))


# print(f"Matrx equivalent = {np.polyval(gE2_list,x2_real)}")
print("i_lim_estimate = ", value(i_lim_estimate), pyunits.get_units(i_lim_estimate))
print(
    f"non_dim_field = {non_dim_field }",
    "Electric field at enhancement = "
    f"{value(field_generated):.3e}"
    f", {pyunits.get_units(field_generated)}",
)
print(
    f"Depletion length = {value(lambda_depletion):.3e}, {pyunits.get_units(lambda_depletion)}"
)
print(
    f"Potential drop = {value(field_generated * lambda_depletion):.3e},"
    f" {pyunits.get_units(field_generated * lambda_depletion)}"
)
# 8.192684904017142
# 36.63880072723734

relative_permittivity = 30

potential_membrane_bpem = 0.8 * pyunits.volt

print(
    f" Applied potential drop = {value(potential_membrane_bpem):.3e}, {pyunits.get_units(potential_membrane_bpem)}"
)

print(
    f" Membrane fixed charge = {value(membrane_fixed_charge):.3e}, {pyunits.get_units(membrane_fixed_charge)}"
)

# elec_field_non_dim=temperature ** -2 * const*sqrt(potential_membrane_bpem/(Constants.vacuum_electric_permittivity *
#                                        relative_permittivity/ (Constants.faraday_constant * membrane_fixed_charge)))/relative_permittivity

# elec_dim_first = sqrt(potential_membrane_bpem * Constants.faraday_constant * membrane_fixed_charge/
#                       (Constants.vacuum_electric_permittivity *relative_permittivity))
# print(f" dimensional elec field original  = {value(elec_dim_first):.3e}, {pyunits.get_units(elec_dim_first)}")

# depletion_length = sqrt( potential_membrane_bpem
#     * Constants.vacuum_electric_permittivity
#     * relative_permittivity
#     / (
#             Constants.faraday_constant
#             * membrane_fixed_charge
#     )
# )

# elec_dim_second = potential_membrane_bpem / depletion_length
#
# print(f" dimensional elec field second  = {value(elec_dim_second):.3e}, {pyunits.get_units(elec_dim_second)}")


depletion_length_corrected = sqrt(
    (
        potential_membrane_bpem
        + Constants.gas_constant
        * temperature
        * log(
            membrane_fixed_charge
            / pyunits.convert(conc_salt, to_units=pyunits.mole * pyunits.meter**-3)
        )
        / Constants.faraday_constant
    )
    * Constants.vacuum_electric_permittivity
    * relative_permittivity
    / (Constants.faraday_constant * membrane_fixed_charge)
)
# print(f"Depletion length corrected = {value(depletion_length_corrected):.3e}, {pyunits.get_units(depletion_length_corrected)}")
# elec_dim_third = potential_membrane_bpem / depletion_length_corrected
# print(f" dimensional elec field third  = {value(elec_dim_third):.3e}, {pyunits.get_units(elec_dim_third)}")
#
# print(f" -----inputs----- membrane_fixed_charge = {value(membrane_fixed_charge):.3e}, {pyunits.get_units(membrane_fixed_charge)}, \n depletion_length_corrected = {value(depletion_length_corrected):.3e}, {pyunits.get_units(depletion_length_corrected)},\n relative permittivity = {value(relative_permittivity):.3e}, {pyunits.get_units(relative_permittivity)}")

elec_dim_fourth = (
    membrane_fixed_charge
    * Constants.faraday_constant
    * depletion_length_corrected
    / (Constants.vacuum_electric_permittivity * relative_permittivity)
)

print(
    f" dimensional elec field  = {value(elec_dim_fourth):.3e}, {pyunits.get_units(elec_dim_fourth)}"
)

print(
    f" dimensional elec field ratio = {value(elec_dim_fourth/field_generated):.3e}, {pyunits.get_units(elec_dim_fourth/field_generated)}"
)

elec_field_non_dim = elec_dim_fourth * const / (relative_permittivity * temperature**2)

# elec_dim_input = 5 * 10 **8 * pyunits.volts * pyunits.meter ** -1

# elec_field_non_dim = elec_dim_input * const /(relative_permittivity * temperature **2)

print(
    f" Non dimensional elec field  = {value(elec_field_non_dim):.3e}, {pyunits.get_units(elec_field_non_dim)}"
)

# print(f" Non dim elec field  = {value(elec_field_non_dim):.3e}, {pyunits.get_units(elec_field_non_dim)}")
#
# elec_dim=const **-1*elec_field_non_dim *temperature ** 2 *relative_permittivity **1
#
# print(f" dimensional elec field  = {value(elec_dim):.3e}, {pyunits.get_units(elec_dim)}")

# elec_field_non_dim = 8.192684904017142 * pyunits.dimensionless
k2_zero = 2 * 10**-6 * pyunits.second**-1
conc_water_deux = 50 * 1e3 * pyunits.mol * pyunits.meter**-3
membrane_fixed_catalyst = 5e3 * pyunits.mole * pyunits.meter**-3
relative_permittivity = 30 * pyunits.dimensionless
membrane_fixed_charge = 5e3 * pyunits.mole * pyunits.meter**-3
terms = 20
matrx = 0
k_a = 5e3 / catalyst_correction * pyunits.mole * pyunits.meter**-3

for indx in range(terms):

    # print("\n index = ", indx, "... denominator:", (math.factorial(indx) * math.factorial(indx + 1))/2 ** indx )

    matrx += (
        2**indx
        * elec_field_non_dim**indx
        / (math.factorial(indx) * math.factorial(indx + 1))
    )


fe_val = matrx

fe_approx = (
    sqrt(2 / math.pi)
    * (8 * value(non_dim_field)) ** (-3 / 4)
    * math.exp((8 * value(non_dim_field)) ** 0.5)
)

print(
    f" f(E)  = {value(fe_val):.3e}, {pyunits.get_units(fe_val)}, Approximate f(E) = {value(fe_approx)}"
)

print(f" k2(E)  = {value(fe_val*k2_zero):.3e}, {pyunits.get_units(fe_val*k2_zero)}")

print(
    f" production  = {value(fe_val*k2_zero*conc_water_deux):.3e}, {pyunits.get_units(fe_val*k2_zero*conc_water_deux)}"
)

print(f" reassociation  = {value(kr_loc * kw):.3e}, {pyunits.get_units(kr_loc * kw)}")

matrx *= k2_zero * conc_water_deux * membrane_fixed_catalyst / k_a

print(f" Rate  = {value(matrx):.3e}, {pyunits.get_units(matrx)}")


matrx *= depletion_length_corrected


print(f" flux  = {value(matrx):.3e}, {pyunits.get_units(matrx)}")

print(
    f" current  = {value(matrx * Constants.faraday_constant):.3e}, {pyunits.get_units(matrx * Constants.faraday_constant)}"
)

print(
    f" depletion length catalyst  ={value(depletion_length_corrected):.3e}, {pyunits.get_units(depletion_length_corrected)}"
)

print(
    f" Total current  = {value(matrx * Constants.faraday_constant + i_lim_estimate):.3e}, {pyunits.get_units(matrx * Constants.faraday_constant+ i_lim_estimate)}"
)
# elec_field_non_dim_scale = 8.192684904017142 **-1
# k2_zero_scale  = (2 * 10**-6) **-1
# conc_water_deux_scale  = (50 * 1e3)**-1
# membrane_fixed_catalyst_scale  = (5e3) **-1
# relative_permittivity_scale  =30 **-1
# membrane_fixed_charge_scale  = (1.5e3) **-1
# terms = 40
# matrx_scale = 0
# k_a_scale  = (1e-6) **-1
# potential_membrane_bpem_scale = 0.1 **-1

# elec_field_non_dim_scale = 1e-1
# k2_zero_scale  = 1e5
# conc_water_deux_scale  = 1e-4
# membrane_fixed_catalyst_scale  = 1e-3
# relative_permittivity_scale =1e-1
# membrane_fixed_charge_scale  = 1e-3
# terms = 40
# matrx_scale = 0
# k_a_scale  = 1e-4
# potential_membrane_bpem_scale = 1e1
# for indx in range(terms):
#
#     matrx_scale += (2 ** indx * elec_field_non_dim_scale ** -indx
#             / (math.factorial(indx) * math.factorial(indx + 1)) )
#
# matrx_scale **=-1
# print(
#     f" f(E) scale = {value(matrx_scale):.3e}, {pyunits.get_units(matrx_scale)}"
# )
# fe_val_scale = matrx_scale
#
# matrx_scale *= (k2_zero_scale * conc_water_deux_scale
#           * membrane_fixed_catalyst_scale/k_a_scale)
#
# print(
#     f" Rate scale = {value(matrx_scale):.3e}, {pyunits.get_units(matrx_scale)}"
# )
#
# matrx_scale *= sqrt(potential_membrane_bpem_scale
#     * value(Constants.vacuum_electric_permittivity)**-1
#     * relative_permittivity_scale
#     / (
#             value(Constants.faraday_constant)**-1
#             * membrane_fixed_charge_scale
#     )
# )
#
# print(
#     f" flux scale = {value(matrx_scale):.3e}, {pyunits.get_units(matrx_scale)}"
# )
#
# print(
#     f" f(E) scaling estimate = {value(fe_val_scale)*value(fe_val):.3e}"
# )
