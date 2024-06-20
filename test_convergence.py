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


kw = 10**-8 * pyunits.mol**2 * pyunits.meter**-6
kd_zero = 2.5 * 10**5 * pyunits.second**-1
kr = 1.33 * 10**11 * pyunits.L * pyunits.mole**-1 * pyunits.second**-1
conc_water = 55 * pyunits.mol * pyunits.L**-1
current_dens_lim_bpem = 1 * pyunits.Amp * pyunits.meter**-2
membrane_fixed_charge = 1.5e3 * pyunits.mol * pyunits.meter**-3
relative_permittivity = 20
temperature = 298

kr_loc = pyunits.convert(
    kr, to_units=pyunits.meter**3 * pyunits.mole**-1 * pyunits.second**-1
)
conc_water_loc = pyunits.convert(conc_water, to_units=pyunits.mol * pyunits.meter**-3)
frac = 1 * 10**-1
const = 0.0936 * pyunits.K**2 * pyunits.volt**-1 * pyunits.meter


qty_une = (
    Constants.vacuum_electric_permittivity
    * relative_permittivity**2
    * temperature**2
    * Constants.avogadro_number
    * Constants.elemental_charge
) / (const * Constants.faraday_constant * membrane_fixed_charge)
qty_deux = kd_zero * conc_water_loc
qty_trois = kr_loc * kw
y2 = current_dens_lim_bpem * frac

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
print(
    f"non_dim_field - 1 = {non_dim_field - 1}",
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
