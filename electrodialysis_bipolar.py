#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
import math

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

from watertap.core import ControlVolume0DBlock, InitializationMixin
from watertap.costing.unit_models.electrodialysis import cost_electrodialysis

__author__ = " Johnson Dhanasekaran"

_log = idaeslog.getLogger(__name__)


class LimitingCurrentDensityMethod(Enum):
    InitialValue = 0
    Empirical = 1
    # Theoretical = 2


class LimitingpotentialMethod(Enum):
    InitialValue = 0
    Empirical = 1
    # Theoretical = 2


class ElectricalOperationMode(Enum):
    Constant_Current = 0
    Constant_Voltage = 1


class LimitingOperationMode(Enum):
    Sub_limiting = 0
    Over_limiting = 1


class PressureDropMethod(Enum):
    none = 0
    experimental = 1
    Darcy_Weisbach = 2


class FrictionFactorMethod(Enum):
    fixed = 0
    Gurreri = 1
    Kuroda = 2


class HydraulicDiameterMethod(Enum):
    fixed = 0
    spacer_specific_area_known = 1
    conventional = 2


# Name of the unit model
@declare_process_block_class("BiPolarElectrodialysis0D")
class Electrodialysis0DData(InitializationMixin, UnitModelBlockData):
    """
    0D BiPolarElectrodialysis Model
    """

    # CONFIG are options for the unit model
    CONFIG = ConfigBlock()  #

    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be False",
            doc="""Indicates whether this model will be dynamic or not,
    **default** = False. The filtration unit does not support dynamic
    behavior, thus this must be False.""",
        ),
    )

    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be False",
            doc="""Indicates whether holdup terms should be constructed or not.
    **default** - False. The filtration unit does not have defined volume, thus
    this must be False.""",
        ),
    )
    CONFIG.declare(
        "has_pressure_change",
        ConfigValue(
            default=False,
            domain=In([True, False]),
            description="Pressure change term construction flag",
            doc="""Indicates whether terms for pressure change should be
    constructed,
    **default** - False.
    **Valid values:** {
    **True** - include pressure change terms,
    **False** - exclude pressure change terms.}""",
        ),
    )
    CONFIG.declare(
        "pressure_drop_method",
        ConfigValue(
            default=PressureDropMethod.none,
            domain=In(PressureDropMethod),
            description="Method to calculate the frictional pressure drop in electrodialysis channels",
            doc="""
     **default** - ``PressureDropMethod.none``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``PressureDropMethod.none``", "The frictional pressure drop is neglected." 
           "``PressureDropMethod.experimental``", "The pressure drop is calculated by an experimental data as pressure drop per unit lenght."
           "``PressureDropMethod.Darcy_Weisbach``", "The pressure drop is calculated by the Darcy-Weisbach equation."
       """,
        ),
    )
    CONFIG.declare(
        "friction_factor_method",
        ConfigValue(
            default=FrictionFactorMethod.fixed,
            domain=In(FrictionFactorMethod),
            description="Method to calculate the Darcy's friction factor",
            doc="""
     **default** - ``FrictionFactorMethod.fixed``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``FrictionFactorMethod.fixed``", "Friction factor is fixed by users" 
           "``FrictionFactorMethod.Gurreri``", "Friction factor evaluated based on Gurreri's work"
           "``FrictionFactorMethod.Kuroda``", "Friction factor evaluated based on Kuroda's work"
       """,
        ),
    )

    CONFIG.declare(
        "hydraulic_diameter_method",
        ConfigValue(
            default=HydraulicDiameterMethod.conventional,
            domain=In(HydraulicDiameterMethod),
            description="Method to calculate the hydraulic diameter for a rectangular channel in ED",
            doc="""
     **default** - ``HydraulicDiameterMethod.conventional``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``HydraulicDiameterMethod.fixed``", "Hydraulic diameter is fixed by users" 
           "``HydraulicDiameterMethod.conventional``", "Conventional method for a rectangular channel with spacer porosity considered" 
           "``HydraulicDiameterMethod.spacer_specific_area_known``", "A method for spacer-filled channel requiring the spacer specific area data"
       """,
        ),
    )

    CONFIG.declare(
        "operation_mode",
        ConfigValue(
            default=ElectricalOperationMode.Constant_Current,
            domain=In(ElectricalOperationMode),
            description="The electrical operation mode. To be selected between Constant Current and Constant Voltage",
        ),
    )

    CONFIG.declare(
        "limiting_current_density_method_bpem",
        ConfigValue(
            default=LimitingCurrentDensityMethod.InitialValue,
            domain=In(LimitingCurrentDensityMethod),
            description="Configuration for method to compute the limiting current density",
            doc="""
           **default** - ``LimitingCurrentDensityMethod.InitialValue``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``LimitingCurrentDensityMethod.InitialValue``", "Limiting current is calculated from a single initial value given by the user."
           "``LimitingCurrentDensityMethod.Empirical``", "Limiting current density is calculated from the empirical equation"
       """,
        ),
    )
    CONFIG.declare(
        "Operation_method_bpem",
        ConfigValue(
            default=LimitingOperationMode.Sub_limiting,
            domain=In(LimitingOperationMode),
            description="Configuration for method of operation",
            doc="""
               **default** - ``LimitingOperationMode.Sub_limiting``

           .. csv-table::
               :header: "Configuration Options", "Description"

               "``LimitingOperationMode.Sub_limiting``", "Salt transport below limiting current. No water splitting"
               "``LimitingOperationMode.Over_limiting``", "Above salt limiting current. Water splitting."
           """,
        ),
    )

    CONFIG.declare(
        "limiting_potential_method_bpem",
        ConfigValue(
            default=LimitingpotentialMethod.InitialValue,
            domain=In(LimitingpotentialMethod),
            description="Configuration for method to compute the limiting potential",
            doc="""
               **default** - ``LimitingpotentialMethod.InitialValue``

           .. csv-table::
               :header: "Configuration Options", "Description"

               "``LimitingpotentialMethod.InitialValue``", "Limiting current is calculated from a initial value given by the user."
               "``LimitingpotentialMethod.Empirical``", "Limiting current density is caculated from the empirical equation"
           """,
        ),
    )

    CONFIG.declare(
        "limiting_current_density_data",
        ConfigValue(
            default=0.5,
            description="Limiting current density data input",
        ),
    )

    CONFIG.declare(
        "limiting_potential_data",
        ConfigValue(
            default=0.5,
            description="Limiting potential data input",
        ),
    )

    CONFIG.declare(
        "material_balance_type",
        ConfigValue(
            default=MaterialBalanceType.useDefault,
            domain=In(MaterialBalanceType),
            description="Material balance construction flag",
            doc="""Indicates what type of mass balance should be constructed,
    **default** - MaterialBalanceType.useDefault.
    **Valid values:** {
    **MaterialBalanceType.useDefault - refer to property package for default
    balance type
    **MaterialBalanceType.none** - exclude material balances,
    **MaterialBalanceType.componentPhase** - use phase component balances,
    **MaterialBalanceType.componentTotal** - use total component balances,
    **MaterialBalanceType.elementTotal** - use total element balances,
    **MaterialBalanceType.total** - use total material balance.}""",
        ),
    )

    CONFIG.declare(
        "is_isothermal",
        ConfigValue(
            default=True,
            domain=Bool,
            description="""Assume isothermal conditions for control volume(s); energy_balance_type must be EnergyBalanceType.none,
    **default** - True.""",
        ),
    )

    CONFIG.declare(
        "energy_balance_type",
        ConfigValue(
            default=EnergyBalanceType.none,
            domain=In(EnergyBalanceType),
            description="Energy balance construction flag",
            doc="""Indicates what type of energy balance should be constructed,
    **default** - EnergyBalanceType.none.
    **Valid values:** {
    **EnergyBalanceType.useDefault - refer to property package for default
    balance type
    **EnergyBalanceType.none** - exclude energy balances,
    **EnergyBalanceType.enthalpyTotal** - single enthalpy balance for material,
    **EnergyBalanceType.enthalpyPhase** - enthalpy balances for each phase,
    **EnergyBalanceType.energyTotal** - single energy balance for material,
    **EnergyBalanceType.energyPhase** - energy balances for each phase.}""",
        ),
    )

    CONFIG.declare(
        "momentum_balance_type",
        ConfigValue(
            default=MomentumBalanceType.pressureTotal,
            domain=In(MomentumBalanceType),
            description="Momentum balance construction flag",
            doc="""Indicates what type of momentum balance should be constructed,
    **default** - MomentumBalanceType.pressureTotal.
    **Valid values:** {
    **MomentumBalanceType.none** - exclude momentum balances,
    **MomentumBalanceType.pressureTotal** - single pressure balance for material,
    **MomentumBalanceType.pressurePhase** - pressure balances for each phase,
    **MomentumBalanceType.momentumTotal** - single momentum balance for material,
    **MomentumBalanceType.momentumPhase** - momentum balances for each phase.}""",
        ),
    )

    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property calculations,
    **default** - useDefault.
    **Valid values:** {
    **useDefault** - use default package from parent model or flowsheet,
    **PhysicalParameterObject** - a PhysicalParameterBlock object.}""",
        ),
    )

    CONFIG.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
    and used when constructing these,
    **default** - None.
    **Valid values:** {
    see property package for documentation.}""",
        ),
    )

    def _validate_config(self):
        if (
            self.config.is_isothermal
            and self.config.energy_balance_type != EnergyBalanceType.none
        ):
            raise ConfigurationError(
                "If the isothermal assumption is used then the energy balance type must be none"
            )

    def build(self):
        # build always starts by calling super().build()
        # This triggers a lot of boilerplate in the background for you
        super().build()
        # this creates blank scaling factors, which are populated later
        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # Check configs for errors
        self._validate_config()

        # Create essential sets.
        self.membrane_set = Set(initialize=["bpem"])
        self.electrode_side = Set(initialize=["cathode_left", "anode_right"])
        add_object_reference(self, "ion_set", self.config.property_package.ion_set)

        add_object_reference(
            self, "cation_set", self.config.property_package.cation_set
        )
        add_object_reference(self, "anion_set", self.config.property_package.anion_set)
        # Create unit model parameters and vars
        self.water_density = Param(
            initialize=1000,
            units=pyunits.kg * pyunits.m**-3,
            doc="density of water",
        )

        self.cell_pair_num = Var(
            initialize=1,
            domain=NonNegativeReals,
            bounds=(1, 10000),
            units=pyunits.dimensionless,
            doc="cell pair number in a stack",
        )

        # electrodialysis cell dimensional properties
        self.cell_width = Var(
            initialize=0.1,
            bounds=(1e-3, 1e2),
            units=pyunits.meter,
            doc="The width of the electrodialysis cell, denoted as b in the model description",
        )
        self.cell_length = Var(
            initialize=0.5,
            bounds=(1e-3, 1e2),
            units=pyunits.meter,
            doc="The length of the electrodialysis cell, denoted as l in the model description",
        )
        self.channel_height = Var(
            initialize=0.0001,
            units=pyunits.meter,
            doc="The distance between the consecutive aem and cem",
        )
        self.spacer_porosity = Var(
            initialize=0.7,
            bounds=(0.01, 1),
            units=pyunits.dimensionless,
            doc='The prosity of spacer in the ED channels. This is also referred to elsewhere as "void fraction" or "volume parameters"',
        )

        # Material and Operational properties

        self.membrane_thickness = Var(
            self.membrane_set,
            initialize=1.3e-4,
            bounds=(1e-8, 1e-1),
            units=pyunits.meter,
            doc="Membrane thickness",
        )

        self.solute_diffusivity_membrane = Var(
            self.membrane_set,
            self.ion_set | self.config.property_package.solute_set,
            initialize=1e-10,
            bounds=(0.0, 1e-6),
            units=pyunits.meter**2 * pyunits.second**-1,
            doc="Solute (ionic and neutral) diffusivity in the membrane phase",
        )
        self.ion_trans_number_membrane = Var(
            self.membrane_set,
            self.ion_set,
            initialize=0.5,
            bounds=(0, 1),
            units=pyunits.dimensionless,
            doc="Ion transference number in the membrane phase",
        )
        self.water_trans_number_membrane = Var(
            self.membrane_set,
            initialize=5,
            bounds=(0, 50),
            units=pyunits.dimensionless,
            doc="Transference number of water in membranes",
        )
        self.water_permeability_membrane = Var(
            self.membrane_set,
            initialize=1e-14,
            units=pyunits.meter * pyunits.second**-1 * pyunits.pascal**-1,
            doc="Water permeability coefficient",
        )
        self.membrane_areal_resistance = Var(
            self.membrane_set,
            initialize=2e-4,
            bounds=(1e-6, 1),
            units=pyunits.ohm * pyunits.meter**2,
            doc="Surface resistance of membrane",
        )
        self.electrodes_resistance = Var(
            initialize=0,
            bounds=(0, 100),
            domain=NonNegativeReals,
            units=pyunits.ohm * pyunits.meter**2,
            doc="areal resistance of TWO electrode compartments of a stack",
        )

        self.current = Var(
            self.flowsheet().time,
            initialize=1,
            bounds=(-1e5, 1e5),
            units=pyunits.amp,
            doc="Current across a cell-pair or stack",
        )
        self.voltage = Var(
            self.flowsheet().time,
            initialize=5,
            bounds=(-1e5, 1e5),
            units=pyunits.volt,
            doc="Voltage across a stack, declared under the 'Constant Voltage' mode only",
        )
        self.current_utilization = Var(
            initialize=1,
            bounds=(0, 1),
            units=pyunits.dimensionless,
            doc="The current utilization including water electro-osmosis and ion diffusion",
        )

        # Performance metrics
        self.current_efficiency = Var(
            self.flowsheet().time,
            initialize=0.9,
            bounds=(0, 1),
            units=pyunits.dimensionless,
            doc="The overall current efficiency for deionizaiton",
        )
        self.power_electrical = Var(
            self.flowsheet().time,
            initialize=1,
            bounds=(0, 12100),
            domain=NonNegativeReals,
            units=pyunits.watt,
            doc="Electrical power consumption of a stack",
        )
        self.specific_power_electrical = Var(
            self.flowsheet().time,
            initialize=10,
            bounds=(0, 1000),
            domain=NonNegativeReals,
            units=pyunits.kW * pyunits.hour * pyunits.meter**-3,
            doc="aem side of the bipolar membrane channel-volume-flow-rate-specific electrical power consumption",
        )
        self.recovery_mass_H2O = Var(
            self.flowsheet().time,
            initialize=0.5,
            bounds=(0, 1),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="water recovery ratio calculated by mass",
        )
        self.velocity_aem_side = Var(
            self.flowsheet().time,
            initialize=0.01,
            units=pyunits.meter * pyunits.second**-1,
            doc="Linear velocity of flow in the aem side of the bipolar membrane",
        )
        self.velocity_cem_side = Var(
            self.flowsheet().time,
            initialize=0.01,
            units=pyunits.meter * pyunits.second**-1,
            doc="Linear velocity of flow in the cem side of the bipolar membrane",
        )

        # Fluxes Vars for constructing mass transfer terms
        self.elec_migration_flux_in = Var(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter**-2 * pyunits.second**-1,
            doc="Molar flux_in of a component across the membrane driven by electrical migration",
        )
        self.elec_migration_flux_out = Var(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter**-2 * pyunits.second**-1,
            doc="Molar flux_out of a component across the membrane driven by electrical migration",
        )
        self.nonelec_flux_in = Var(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter**-2 * pyunits.second**-1,
            doc="Molar flux_in of a component across the membrane driven by non-electrical forces",
        )
        self.nonelec_flux_out = Var(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter**-2 * pyunits.second**-1,
            doc="Molar flux_out of a component across the membrane driven by non-electrical forces",
        )

        # Build control volume for the aem side of the bipolar channel
        self.aem_side = ControlVolume0DBlock(
            dynamic=False,
            has_holdup=False,
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
        )
        self.aem_side.add_state_blocks(has_phase_equilibrium=False)
        self.aem_side.add_material_balances(
            balance_type=self.config.material_balance_type, has_mass_transfer=True
        )
        self.aem_side.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_enthalpy_transfer=False,
        )

        if self.config.is_isothermal:
            self.aem_side.add_isothermal_assumption()
        self.aem_side.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=self.config.has_pressure_change,
        )
        # den_mass and visc_d in aem side of the bipolar membrane
        # and cem side of the bipolar membrane channels are the same
        add_object_reference(
            self, "dens_mass", self.aem_side.properties_in[0].dens_mass_phase["Liq"]
        )
        add_object_reference(
            self, "visc_d", self.aem_side.properties_in[0].visc_d_phase["Liq"]
        )
        # Build control volume for the cem side of the bipolar membrane channel
        self.cem_side = ControlVolume0DBlock(
            dynamic=False,
            has_holdup=False,
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
        )
        self.cem_side.add_state_blocks(has_phase_equilibrium=False)
        self.cem_side.add_material_balances(
            balance_type=self.config.material_balance_type, has_mass_transfer=True
        )
        self.cem_side.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_enthalpy_transfer=False,
        )

        if self.config.is_isothermal:
            self.cem_side.add_isothermal_assumption()
        self.cem_side.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=self.config.has_pressure_change,
        )

        # Add ports (creates inlets and outlets for each channel)
        self.add_inlet_port(name="inlet_aem_side", block=self.aem_side)
        self.add_outlet_port(name="outlet_aem_side", block=self.aem_side)
        self.add_inlet_port(name="inlet_cem_side", block=self.cem_side)
        self.add_outlet_port(name="outlet_cem_side", block=self.cem_side)

        # extension options

        if self.config.Operation_method_bpem == LimitingOperationMode.Over_limiting:
            self._make_limiting_operation()

        if (
            not self.config.pressure_drop_method == PressureDropMethod.none
        ) and self.config.has_pressure_change:
            self._pressure_drop_calculation()

            @self.Constraint(
                self.flowsheet().time,
                doc="Pressure drop expression as calculated by the pressure drop data, "
                "aem side of the bipolar membrane.",
            )
            def eq_deltaP_aem_side(self, t):
                return self.aem_side.deltaP[t] == -self.pressure_drop_total[t]

            @self.Constraint(
                self.flowsheet().time,
                doc="Pressure drop expression as calculated by the pressure drop data,"
                "  cem side of the bipolar membrane.",
            )
            def eq_deltaP_cem_side(self, t):
                return self.cem_side.deltaP[t] == -self.pressure_drop_total[t]

        elif self.config.pressure_drop_method == PressureDropMethod.none and (
            not self.config.has_pressure_change
        ):
            pass
        else:
            raise ConfigurationError(
                "A valid (not none) pressure_drop_method and has_pressure_change being True "
                "must be both used or unused at the same time. "
            )

        # Build Constraints
        @self.Constraint(
            self.flowsheet().time,
            doc="Calculate flow velocity in a single aem side of the bipolar membrane channel,"
            " based on the average of inlet and outlet",
        )
        def eq_get_velocity_aem_side(self, t):
            return self.velocity_aem_side[
                t
            ] * self.cell_width * self.channel_height * self.spacer_porosity * self.cell_pair_num == 0.5 * (
                self.aem_side.properties_in[0].flow_vol_phase["Liq"]
                + self.aem_side.properties_out[0].flow_vol_phase["Liq"]
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Calculate flow velocity in a single cem side of the bipolar membrane channel,"
            " based on the average of inlet and outlet",
        )
        def eq_get_velocity_cem_side(self, t):
            return self.velocity_cem_side[
                t
            ] * self.cell_width * self.channel_height * self.spacer_porosity * self.cell_pair_num == 0.5 * (
                self.cem_side.properties_in[0].flow_vol_phase["Liq"]
                + self.cem_side.properties_out[0].flow_vol_phase["Liq"]
            )

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            doc="Current-Voltage relationship",
        )
        def eq_current_voltage_relation(self, t, p):
            total_areal_resistance = (
                self.membrane_areal_resistance["bpem"]
                + self.channel_height
                * (
                    0.5**-1
                    * (
                        self.cem_side.properties_in[t].elec_cond_phase["Liq"]
                        + self.cem_side.properties_out[t].elec_cond_phase["Liq"]
                    )
                    ** -1
                    + 0.5**-1
                    * (
                        self.aem_side.properties_in[t].elec_cond_phase["Liq"]
                        + self.aem_side.properties_out[t].elec_cond_phase["Liq"]
                    )
                    ** -1
                )
            ) * self.cell_pair_num + self.electrodes_resistance
            if self.config.Operation_method_bpem == LimitingOperationMode.Over_limiting:
                return (
                    self.current[t]
                    * (self.cell_width * self.cell_length) ** -1
                    * total_areal_resistance
                    + self.potential_lim_bpem[t] * self.cell_pair_num
                    == self.voltage[t]
                )
            else:
                return (
                    self.current[t]
                    * (self.cell_width * self.cell_length) ** -1
                    * total_areal_resistance
                    == self.voltage[t]
                )

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for electrical migration flux_in",
        )
        def eq_elec_migration_flux_in(self, t, p, j):
            if j == "H2O":
                return self.elec_migration_flux_in[t, p, j] == (
                    self.water_trans_number_membrane["bpem"]
                ) * (
                    self.current[t]
                    / (self.cell_width * self.cell_length)
                    / Constants.faraday_constant
                )

            elif j in self.ion_set:
                if not (j == "H_+" or j == "OH_-"):
                    if (
                        self.config.Operation_method_bpem
                        == LimitingOperationMode.Sub_limiting
                    ):
                        return self.elec_migration_flux_in[t, p, j] == (
                            self.ion_trans_number_membrane["bpem", j]
                        ) * (
                            self.current_utilization
                            * self.current[t]
                            / (self.cell_width * self.cell_length)
                        ) / (
                            self.config.property_package.charge_comp[j]
                            * Constants.faraday_constant
                        )
                    else:
                        return self.elec_migration_flux_in[t, p, j] == (
                            self.ion_trans_number_membrane["bpem", j]
                        ) * (
                            self.current_utilization
                            * self.current_dens_lim_bpem[t]
                            / (self.cell_width * self.cell_length)
                        ) / (
                            self.config.property_package.charge_comp[j]
                            * Constants.faraday_constant
                        )
                else:
                    if (
                        self.config.Operation_method_bpem
                        == LimitingOperationMode.Over_limiting
                    ):
                        return self.elec_migration_flux_in[t, p, j] == (1) * (
                            1
                            * (self.current[t] - self.current_dens_lim_bpem[t])
                            / (self.cell_width * self.cell_length)
                        ) / (
                            self.config.property_package.charge_comp[j]
                            * Constants.faraday_constant
                        )
                    else:
                        return (
                            self.elec_migration_flux_in[t, p, j]
                            == 0 * pyunits.mol * pyunits.meter**-2 * pyunits.s**-1
                        )

            else:
                self.elec_migration_flux_in[t, p, j].fix(
                    0 * pyunits.mol * pyunits.m**-2 * pyunits.s**-1
                )
                return Constraint.Skip

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for electrical migration flux_out",
        )
        def eq_elec_migration_flux_out(self, t, p, j):
            return (
                self.elec_migration_flux_out[t, p, j]
                == self.elec_migration_flux_in[t, p, j]
            )

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for non-electrical flux_in",
        )
        def eq_nonelec_flux_in(self, t, p, j):
            if j == "H2O":
                return self.nonelec_flux_in[
                    t, p, j
                ] == self.water_density / self.config.property_package.mw_comp[j] * (
                    self.water_permeability_membrane["bpem"]
                ) * (
                    self.cem_side.properties_in[t].pressure_osm_phase[p]
                    - self.aem_side.properties_in[t].pressure_osm_phase[p]
                )

            else:
                return (
                    self.nonelec_flux_in[t, p, j]
                    == 0 * pyunits.mol * pyunits.m**-2 * pyunits.s**-1
                )

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for non-electrical flux_out",
        )
        def eq_nonelec_flux_out(self, t, p, j):
            if j == "H2O":
                return self.nonelec_flux_out[
                    t, p, j
                ] == self.water_density / self.config.property_package.mw_comp[j] * (
                    self.water_permeability_membrane["bpem"]
                ) * (
                    self.cem_side.properties_in[t].pressure_osm_phase[p]
                    - self.aem_side.properties_in[t].pressure_osm_phase[p]
                )

            else:
                return (
                    self.nonelec_flux_out[t, p, j]
                    == 0 * pyunits.mol * pyunits.m**-2 * pyunits.s**-1
                )

        # Add constraints for mass transfer terms (aem side of the bipolar membrane)
        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer term for the aem side of the bipolar membrane",
        )
        def eq_mass_transfer_term_aem_side(self, t, p, j):
            if j == "H_+":
                return (
                    self.aem_side.mass_transfer_term[t, p, j]
                    == 0
                    * pyunits.mol
                    * pyunits.meter**-2
                    * pyunits.s**-1
                    * (self.cell_width * self.cell_length)
                    * self.cell_pair_num
                )
            else:

                return (
                    self.aem_side.mass_transfer_term[t, p, j]
                    == -0.5
                    * (
                        self.elec_migration_flux_in[t, p, j]
                        + self.elec_migration_flux_out[t, p, j]
                        + self.nonelec_flux_in[t, p, j]
                        + self.nonelec_flux_out[t, p, j]
                    )
                    * (self.cell_width * self.cell_length)
                    * self.cell_pair_num
                )

        # Add constraints for mass transfer terms (cem side of the bipolar membrane)
        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer term for the cem side of the bipolar membrane channel",
        )
        def eq_mass_transfer_term_cem_side(self, t, p, j):
            if j == "OH_-":
                return (
                    self.cem_side.mass_transfer_term[t, p, j]
                    == 0
                    * pyunits.mol
                    * pyunits.meter**-2
                    * pyunits.s**-1
                    * (self.cell_width * self.cell_length)
                    * self.cell_pair_num
                )
            else:
                return (
                    self.cem_side.mass_transfer_term[t, p, j]
                    == 0.5
                    * (
                        self.elec_migration_flux_in[t, p, j]
                        + self.elec_migration_flux_out[t, p, j]
                        + self.nonelec_flux_in[t, p, j]
                        + self.nonelec_flux_out[t, p, j]
                    )
                    * (self.cell_width * self.cell_length)
                    * self.cell_pair_num
                )

        @self.Constraint(
            self.flowsheet().time,
            doc="Electrical power consumption of a stack",
        )
        def eq_power_electrical(self, t):
            return self.power_electrical[t] == self.current[t] * self.voltage[t]

        @self.Constraint(
            self.flowsheet().time,
            doc="Aem side of the bipolar membrane_volume_flow_rate_specific electrical power consumption of a stack",
        )
        def eq_specific_power_electrical(self, t):
            return (
                pyunits.convert(
                    self.specific_power_electrical[t],
                    pyunits.watt * pyunits.second * pyunits.meter**-3,
                )
                * self.aem_side.properties_out[t].flow_vol_phase["Liq"]
                == self.current[t] * self.voltage[t]
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Overall current efficiency evaluation",
        )
        def eq_current_efficiency(self, t):
            return (
                self.current_efficiency[t] * self.current[t] * self.cell_pair_num
                == sum(
                    self.aem_side.properties_in[t].flow_mol_phase_comp["Liq", j]
                    * self.config.property_package.charge_comp[j]
                    - self.aem_side.properties_out[t].flow_mol_phase_comp["Liq", j]
                    * self.config.property_package.charge_comp[j]
                    for j in self.config.property_package.cation_set
                )
                * Constants.faraday_constant
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Water recovery by mass",
        )
        def eq_recovery_mass_H2O(self, t):
            return (
                self.recovery_mass_H2O[t]
                * (
                    self.aem_side.properties_in[t].flow_mass_phase_comp["Liq", "H2O"]
                    + self.cem_side.properties_in[t].flow_mass_phase_comp["Liq", "H2O"]
                )
                == self.aem_side.properties_out[t].flow_mass_phase_comp["Liq", "H2O"]
            )

    def _make_limiting_operation(self):
        self.current_dens_lim_bpem = Var(
            self.flowsheet().time,
            initialize=1e2,
            bounds=(0, 1e5),
            units=pyunits.amp * pyunits.meter**-2,
            doc="Limiting current density across the bipolar membrane as a function of the normalized length",
        )

        self.potential_lim_bpem = Var(
            self.flowsheet().time,
            initialize=3.6,
            bounds=(0, 5000),
            units=pyunits.volt,
            doc="Potential drop across the depletion layer",
        )

        self.elec_field_non_dim = Var(
            initialize=1,
            bounds=(0, 1e32),
            units=pyunits.dimensionless,
            doc="Limiting current density across the bipolar membrane as a function of the normalized length",
        )
        self.relative_permittivity = Var(
            self.membrane_set,
            initialize=20,
            bounds=(1, 80),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Relative permittivity",
        )
        self.membrane_fixed_charge = Var(
            self.membrane_set,
            initialize=1.5e3,
            bounds=(1e-1, 1e5),
            units=pyunits.mole * pyunits.meter**-3,
            doc="Membrane fixed charge",
        )
        self.kr = Var(
            self.membrane_set,
            initialize=1.33 * 10**11,
            bounds=(1e-6, 1e16),
            units=pyunits.L * pyunits.mole**-1 * pyunits.second**-1,
            doc="Re-association rate constant",
        )
        self.kd_zero = Var(
            self.membrane_set,
            initialize=2 * 10**-5,
            bounds=(1e-10, 1e2),
            units=pyunits.second**-1,
            doc="Dissociation rate constant at no electric field",
        )
        self.diffus_mass = Var(
            self.membrane_set,
            initialize=2e-9,
            bounds=(1e-16, 1e-6),
            units=pyunits.meter**2 * pyunits.second**-1,
            doc="The mass diffusivity of the solute as molecules (not individual ions)",
        )
        self.salt_conc_aem = Var(
            self.membrane_set,
            initialize=1e3,
            bounds=(1e-6, 1e4),
            units=pyunits.mole * pyunits.meter**-3,
            doc="Salt concentration on the AEM side of the bipolar membrane",
        )
        self.salt_conc_cem = Var(
            self.membrane_set,
            initialize=1e3,
            bounds=(1e-6, 1e4),
            units=pyunits.mole * pyunits.meter**-3,
            doc="Salt concentration on the CEM side of the bipolar membrane",
        )

        @self.Constraint(
            self.flowsheet().time,
            doc="Calculate limiting current density across the bipolar membrane",
        )
        def eq_current_dens_lim_bpem(self, t):
            if (
                self.config.limiting_current_density_method_bpem
                == LimitingCurrentDensityMethod.InitialValue
            ):
                return self.current_dens_lim_bpem[t] == (
                    self.config.limiting_current_density_data
                    * pyunits.amp
                    * pyunits.meter**-2
                )
            elif (
                self.config.limiting_current_density_method_bpem
                == LimitingCurrentDensityMethod.Empirical
            ):

                return self.current_dens_lim_bpem[t] == self.diffus_mass[
                    "bpem"
                ] * Constants.faraday_constant * (
                    self.salt_conc_aem["bpem"] + self.salt_conc_cem["bpem"]
                ) ** 2 / (
                    self.membrane_thickness["bpem"] * self.membrane_fixed_charge["bpem"]
                )

        @self.Constraint(
            self.flowsheet().time,
            doc="Calculate the potential barrier at limiting current across the bipolar membrane",
        )
        def eq_potential_lim_bpem(self, t):
            if self.config.Operation_method_bpem == LimitingOperationMode.Over_limiting:
                if (
                    self.config.limiting_potential_method_bpem
                    == LimitingpotentialMethod.InitialValue
                ):
                    return self.potential_lim_bpem[t] == (
                        self.config.limiting_potential_data * pyunits.volt
                    )

                elif (
                    self.config.limiting_potential_method_bpem
                    == LimitingpotentialMethod.Empirical
                ):
                    #   [H+][OH-] concentration
                    kw = 10**-8 * pyunits.mol**2 * pyunits.meter**-6
                    conc_water = 55 * pyunits.mol * pyunits.L**-1
                    kr_loc = pyunits.convert(
                        self.kr["bpem"],
                        to_units=pyunits.meter**3
                        * pyunits.mole**-1
                        * pyunits.second**-1,
                    )
                    conc_water_loc = pyunits.convert(
                        conc_water, to_units=pyunits.mol * pyunits.meter**-3
                    )
                    # Fraction of threshold of limiting current: currently 0.1 i_lim
                    frac = 1 * 10**-1
                    # Dimensional pre-factor to evaulate non-dimensional electric field
                    const = 0.0936 * pyunits.K**2 * pyunits.volt**-1 * pyunits.meter

                    @self.Constraint(
                        self.flowsheet().time,
                        doc="Calculate the non-dimensional potential drop",
                    )
                    def eq_potential_lim_bpem_non_dim(self, t):
                        # [y2, qty_une, qty_deux, qty_trois] = dat
                        terms = 40
                        matrx = 0
                        for indx in range(terms):
                            # rev_indx = terms - indx - 1
                            matrx += (
                                2**indx
                                * self.elec_field_non_dim**indx
                                / (math.factorial(indx) * math.factorial(indx + 1))
                            )

                        matrx *= self.kd_zero["bpem"] * conc_water_loc
                        matrx += -kr_loc * kw
                        return (
                            Constants.vacuum_electric_permittivity
                            * self.relative_permittivity["bpem"] ** 2
                            * self.aem_side.properties_in[t].temperature ** 2
                            * Constants.avogadro_number
                            * Constants.elemental_charge
                        ) / (
                            const
                            * Constants.faraday_constant
                            * self.membrane_fixed_charge["bpem"]
                        ) * matrx * self.elec_field_non_dim == self.current_dens_lim_bpem[
                            t
                        ] * frac

                    # Dimensional electric field
                    field_generated = (
                        self.elec_field_non_dim
                        * self.relative_permittivity["bpem"]
                        * self.aem_side.properties_in[t].temperature ** 2
                        / const
                    )

                    # Depletion length at the junction of the bipolar membrane
                    lambda_depletion = (
                        field_generated
                        * Constants.vacuum_electric_permittivity
                        * self.relative_permittivity["bpem"]
                        / (
                            Constants.faraday_constant
                            * self.membrane_fixed_charge["bpem"]
                        )
                    )

                    return (
                        self.potential_lim_bpem[t] == field_generated * lambda_depletion
                    )

            else:
                return self.potential_lim_bpem[t] == 0 * pyunits.volt

    def _get_fluid_dimensionless_quantities(self):
        self.hydraulic_diameter = Var(
            initialize=1e-3,
            bounds=(0, None),
            units=pyunits.meter,
            doc="The hydraulic diameter of the channel",
        )
        self.N_Re = Var(
            initialize=50,
            bounds=(0, None),
            units=pyunits.dimensionless,
            doc="Reynolds Number",
        )

        if self.config.hydraulic_diameter_method == HydraulicDiameterMethod.fixed:
            _log.warning("Do not forget to FIX the channel hydraulic diameter in [m]!")
        else:

            @self.Constraint(
                doc="To calculate hydraulic diameter",
            )
            def eq_hydraulic_diameter(self):
                if (
                    self.config.hydraulic_diameter_method
                    == HydraulicDiameterMethod.conventional
                ):
                    return (
                        self.hydraulic_diameter
                        == 2
                        * self.channel_height
                        * self.cell_width
                        * self.spacer_porosity
                        * (self.channel_height + self.cell_width) ** -1
                    )
                else:
                    self.spacer_specific_area = Var(
                        initialize=1e4,
                        bounds=(0, None),
                        units=pyunits.meter**-1,
                        doc="The specific area of the channel",
                    )
                    return (
                        self.hydraulic_diameter
                        == 4
                        * self.spacer_porosity
                        * (
                            2 * self.channel_height**-1
                            + (1 - self.spacer_porosity) * self.spacer_specific_area
                        )
                        ** -1
                    )

        @self.Constraint(
            doc="To calculate Re",
        )
        def eq_Re(self):

            return (
                self.N_Re
                == self.dens_mass
                * self.velocity_aem_side[0]
                * self.hydraulic_diameter
                * self.visc_d**-1
            )

    def _pressure_drop_calculation(self):
        self.pressure_drop_total = Var(
            self.flowsheet().time,
            initialize=1e6,
            units=pyunits.pascal,
            doc="pressure drop over an entire ED stack",
        )
        self.pressure_drop = Var(
            self.flowsheet().time,
            initialize=1e3,
            units=pyunits.pascal * pyunits.meter**-1,
            doc="pressure drop per unit of length",
        )

        if self.config.pressure_drop_method == PressureDropMethod.experimental:
            _log.warning(
                "Do not forget to FIX the experimental pressure drop value in [Pa/m]!"
            )
        else:
            self._get_fluid_dimensionless_quantities()

            self.friction_factor = Var(
                initialize=10,
                bounds=(0, None),
                units=pyunits.dimensionless,
                doc="friction factor of the channel fluid",
            )

            @self.Constraint(
                self.flowsheet().time,
                doc="To calculate pressure drop over an stack",
            )
            def eq_pressure_drop_total(self, t):
                return (
                    self.pressure_drop_total[t]
                    == self.dens_mass
                    * self.friction_factor
                    * self.velocity_aem_side[0] ** 2
                    * 0.5
                    * self.hydraulic_diameter**-1
                    * self.cell_length
                )

            if self.config.friction_factor_method == FrictionFactorMethod.fixed:
                _log.warning("Do not forget to FIX the Darcy's friction factor value!")
            else:

                @self.Constraint(
                    doc="To calculate friction factor",
                )
                def eq_friction_factor(self):
                    if (
                        self.config.friction_factor_method
                        == FrictionFactorMethod.Gurreri
                    ):
                        return (
                            self.friction_factor
                            == 4 * 50.6 * self.spacer_porosity**-7.06 * self.N_Re**-1
                        )
                    elif (
                        self.config.friction_factor_method
                        == FrictionFactorMethod.Kuroda
                    ):
                        return (
                            self.friction_factor
                            == 4 * 9.6 * self.spacer_porosity**-1 * self.N_Re**-0.5
                        )

        @self.Constraint(
            self.flowsheet().time,
            doc="To calculate total pressure drop over a stack",
        )
        def eq_pressure_drop(self, t):
            return (
                self.pressure_drop[t]
                == self.pressure_drop_total[t] * self.cell_length**-1
            )

    # initialize method
    def initialize_build(
        self,
        state_args=None,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        General wrapper for pressure changer initialization routines

        Keyword Arguments:
            state_args : a dict of arguments to be passed to the property
                         package(s) to provide an initial state for
                         initialization (see documentation of the specific
                         property package) (default = {}).
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None)
            solver : str indicating which solver to use during
                     initialization (default = None)

        Returns: None
        """
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")
        # Set solver options
        opt = get_solver(solver, optarg)

        # ---------------------------------------------------------------------

        # Set the outlet has the same intial condition of the inlet.
        for k in self.keys():
            for j in self[k].config.property_package.component_list:
                self[k].aem_side.properties_out[0].flow_mol_phase_comp["Liq", j] = (
                    value(
                        self[k].aem_side.properties_in[0].flow_mol_phase_comp["Liq", j]
                    )
                )
                self[k].cem_side.properties_out[0].flow_mol_phase_comp["Liq", j] = (
                    value(
                        self[k].cem_side.properties_in[0].flow_mol_phase_comp["Liq", j]
                    )
                )
        flags_aem_side = self.aem_side.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
            hold_state=True,
        )
        init_log.info_high("Initialization Step 1 Complete.")
        # ---------------------------------------------------------------------
        # Initialize cem side of the bipolar membrane_side block
        flags_cem_side = self.cem_side.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,  # inlet var
            hold_state=True,
        )
        init_log.info_high("Initialization Step 2 Complete.")
        # ---------------------------------------------------------------------
        # Solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)
        init_log.info_high("Initialization Step 3 {}.".format(idaeslog.condition(res)))
        # ---------------------------------------------------------------------
        # Release state
        self.aem_side.release_state(flags_aem_side, outlvl)
        init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))
        self.cem_side.release_state(flags_cem_side, outlvl)
        init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))

        if not check_optimal_termination(res):
            raise InitializationError(f"Unit model {self.name} failed to initialize")

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
        # Scaling factors that user may setup
        # The users are highly encouraged to provide scaling factors for assessable vars below.
        # Not providing these vars will give a warning.
        if (
            iscale.get_scaling_factor(self.solute_diffusivity_membrane, warning=True)
            is None
        ):
            iscale.set_scaling_factor(self.solute_diffusivity_membrane, 1e10)
        if iscale.get_scaling_factor(self.membrane_thickness, warning=True) is None:
            iscale.set_scaling_factor(self.membrane_thickness, 1e4)
        if (
            iscale.get_scaling_factor(self.water_permeability_membrane, warning=True)
            is None
        ):
            iscale.set_scaling_factor(self.water_permeability_membrane, 1e14)
        if iscale.get_scaling_factor(self.cell_pair_num, warning=True) is None:
            iscale.set_scaling_factor(self.cell_pair_num, 0.1)
        if iscale.get_scaling_factor(self.cell_length, warning=True) is None:
            iscale.set_scaling_factor(self.cell_length, 1e1)
        if iscale.get_scaling_factor(self.cell_width, warning=True) is None:
            iscale.set_scaling_factor(self.cell_width, 1e1)
        if iscale.get_scaling_factor(self.channel_height, warning=True) is None:
            iscale.set_scaling_factor(self.channel_height, 1e4)
        if iscale.get_scaling_factor(self.spacer_porosity, warning=True) is None:
            iscale.set_scaling_factor(self.spacer_porosity, 1)
        if (
            iscale.get_scaling_factor(self.membrane_areal_resistance, warning=True)
            is None
        ):
            iscale.set_scaling_factor(self.membrane_areal_resistance, 1e4)
        if iscale.get_scaling_factor(self.electrodes_resistance, warning=True) is None:
            iscale.set_scaling_factor(self.electrodes_resistance, 1e4)
        if iscale.get_scaling_factor(self.current, warning=True) is None:
            iscale.set_scaling_factor(self.current, 1)
        if iscale.get_scaling_factor(self.voltage, warning=True) is None:
            iscale.set_scaling_factor(self.voltage, 1)

        if self.config.Operation_method_bpem == LimitingOperationMode.Over_limiting:

            if iscale.get_scaling_factor(self.potential_lim_bpem, warning=True) is None:
                iscale.set_scaling_factor(self.potential_lim_bpem, 1)
            if (
                iscale.get_scaling_factor(self.membrane_fixed_charge, warning=True)
                is None
            ):
                iscale.set_scaling_factor(self.membrane_fixed_charge, 1e-3)
            if hasattr(self, "diffus_mass") and (
                iscale.get_scaling_factor(self.diffus_mass, warning=True) is None
            ):
                iscale.set_scaling_factor(self.diffus_mass, 1e9)
            if hasattr(self, "salt_conc_aem") and (
                iscale.get_scaling_factor(self.salt_conc_aem, warning=True) is None
            ):
                iscale.set_scaling_factor(self.salt_conc_aem, 1e-2)
            if hasattr(self, "salt_conc_cem") and (
                iscale.get_scaling_factor(self.salt_conc_cem, warning=True) is None
            ):
                iscale.set_scaling_factor(self.salt_conc_cem, 1e-2)
            if iscale.get_scaling_factor(self.kr, warning=True) is None:
                iscale.set_scaling_factor(self.kr, 1e-11)
            if iscale.get_scaling_factor(self.kd_zero, warning=True) is None:
                iscale.set_scaling_factor(self.kd_zero, 1e5)

        # The following Vars are built for constructing constraints and their sf are computed from other Vars.
        iscale.set_scaling_factor(
            self.elec_migration_flux_in,
            iscale.get_scaling_factor(self.current)
            * iscale.get_scaling_factor(self.cell_length) ** -1
            * iscale.get_scaling_factor(self.cell_width) ** -1
            * 1e5,
        )
        iscale.set_scaling_factor(
            self.elec_migration_flux_out,
            iscale.get_scaling_factor(self.current)
            * iscale.get_scaling_factor(self.cell_length) ** -1
            * iscale.get_scaling_factor(self.cell_width) ** -1
            * 1e5,
        )

        for ind in self.nonelec_flux_in:
            if ind[2] == "H2O":
                sf = (
                    1e-3
                    * 0.018
                    * iscale.get_scaling_factor(self.water_permeability_membrane)
                    * iscale.get_scaling_factor(
                        self.cem_side.properties_in[ind[0]].pressure_osm_phase[ind[1]]
                    )
                )
            else:
                sf = 1
            iscale.set_scaling_factor(self.nonelec_flux_in[ind], sf)
        for ind in self.nonelec_flux_out:
            if ind[2] == "H2O":
                sf = (
                    1e-3
                    * 0.018
                    * iscale.get_scaling_factor(self.water_permeability_membrane)
                    * iscale.get_scaling_factor(
                        self.aem_side.properties_in[ind[0]].pressure_osm_phase[ind[1]]
                    )
                )
            else:
                sf = 1

            iscale.set_scaling_factor(self.nonelec_flux_out[ind], sf)
        iscale.set_scaling_factor(
            self.power_electrical,
            iscale.get_scaling_factor(self.current)
            * iscale.get_scaling_factor(self.voltage),
        )
        for ind, c in self.specific_power_electrical.items():
            iscale.set_scaling_factor(
                self.specific_power_electrical[ind],
                3.6e6
                * iscale.get_scaling_factor(self.current[ind])
                * iscale.get_scaling_factor(self.voltage[ind])
                * iscale.get_scaling_factor(
                    self.aem_side.properties_out[ind].flow_vol_phase["Liq"]
                )
                ** -1,
            )
        for ind in self.velocity_aem_side:
            if (
                iscale.get_scaling_factor(self.velocity_aem_side[ind], warning=False)
                is None
            ):
                sf = (
                    iscale.get_scaling_factor(
                        self.aem_side.properties_in[ind].flow_vol_phase["Liq"]
                    )
                    * iscale.get_scaling_factor(self.cell_width) ** -1
                    * iscale.get_scaling_factor(self.channel_height) ** -1
                    * iscale.get_scaling_factor(self.spacer_porosity) ** -1
                    * iscale.get_scaling_factor(self.cell_pair_num) ** -1
                )

                iscale.set_scaling_factor(self.velocity_aem_side[ind], sf)
                iscale.set_scaling_factor(self.velocity_cem_side[ind], sf)
        if hasattr(self, "spacer_specific_area") and (
            iscale.get_scaling_factor(self.spacer_specific_area, warning=True) is None
        ):
            iscale.set_scaling_factor(self.spacer_specific_area, 1e-4)
        if hasattr(self, "hydraulic_diameter") and (
            iscale.get_scaling_factor(self.hydraulic_diameter, warning=True) is None
        ):
            iscale.set_scaling_factor(self.hydraulic_diameter, 1e4)
        if hasattr(self, "N_Re") and (
            iscale.get_scaling_factor(self.N_Re, warning=True) is None
        ):
            sf = (
                iscale.get_scaling_factor(self.dens_mass)
                * iscale.get_scaling_factor(self.velocity_aem_side[0])
                * iscale.get_scaling_factor(self.hydraulic_diameter)
                * iscale.get_scaling_factor(self.visc_d) ** -1
            )
            iscale.set_scaling_factor(self.N_Re, sf)

        if hasattr(self, "friction_factor") and (
            iscale.get_scaling_factor(self.friction_factor, warning=True) is None
        ):
            if self.config.friction_factor_method == FrictionFactorMethod.fixed:
                sf = 0.1
            elif self.config.friction_factor_method == FrictionFactorMethod.Gurreri:
                sf = (
                    (4 * 50.6) ** -1
                    * (iscale.get_scaling_factor(self.spacer_porosity)) ** -7.06
                    * iscale.get_scaling_factor(self.N_Re) ** -1
                )
            elif self.config.friction_factor_method == FrictionFactorMethod.Kuroda:
                sf = (4 * 9.6) ** -1 * iscale.get_scaling_factor(self.N_Re) ** -0.5
            iscale.set_scaling_factor(self.friction_factor, sf)

        if hasattr(self, "pressure_drop") and (
            iscale.get_scaling_factor(self.pressure_drop, warning=True) is None
        ):
            if self.config.pressure_drop_method == PressureDropMethod.experimental:
                sf = 1e-5
            else:
                sf = (
                    iscale.get_scaling_factor(self.dens_mass)
                    * iscale.get_scaling_factor(self.friction_factor)
                    * iscale.get_scaling_factor(self.velocity_aem_side[0]) ** 2
                    * 2
                    * iscale.get_scaling_factor(self.hydraulic_diameter) ** -1
                )
            iscale.set_scaling_factor(self.pressure_drop, sf)

        if hasattr(self, "pressure_drop_total") and (
            iscale.get_scaling_factor(self.pressure_drop_total, warning=True) is None
        ):
            if self.config.pressure_drop_method == PressureDropMethod.experimental:
                sf = 1e-5 * iscale.get_scaling_factor(self.cell_length)
            else:
                sf = (
                    iscale.get_scaling_factor(self.dens_mass)
                    * iscale.get_scaling_factor(self.friction_factor)
                    * iscale.get_scaling_factor(self.velocity_aem_side[0]) ** 2
                    * 2
                    * iscale.get_scaling_factor(self.hydraulic_diameter) ** -1
                    * iscale.get_scaling_factor(self.cell_length)
                )
            iscale.set_scaling_factor(self.pressure_drop_total, sf)

        if hasattr(self, "current_dens_lim_bpem"):
            if iscale.get_scaling_factor(self.current_dens_lim_bpem) is None:
                if (
                    self.config.limiting_current_density_method_bpem
                    == LimitingCurrentDensityMethod.InitialValue
                ):
                    sf = self.config.limiting_current_density_data**-1
                    iscale.set_scaling_factor(self.current_dens_lim_bpem, sf)
                elif (
                    self.config.limiting_current_density_method_bpem
                    == LimitingCurrentDensityMethod.Empirical
                ) and (
                    self.config.Operation_method_bpem
                    == LimitingOperationMode.Over_limiting
                ):
                    sf = (
                        iscale.get_scaling_factor(self.diffus_mass)
                        * value(Constants.faraday_constant) ** -1
                        * (2 * iscale.get_scaling_factor(self.salt_conc_aem)) ** 2
                        / (
                            iscale.get_scaling_factor(self.membrane_thickness)
                            * iscale.get_scaling_factor(self.membrane_fixed_charge)
                        )
                    )

                    print("current_dens_lim_bpem sf:", sf)
                    iscale.set_scaling_factor(self.current_dens_lim_bpem, sf)

        # Constraint scaling
        for ind, c in self.eq_current_voltage_relation.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.membrane_areal_resistance)
            )
        for ind, c in self.eq_power_electrical.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.power_electrical)
            )
        for ind, c in self.eq_specific_power_electrical.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.specific_power_electrical[ind])
            )

        for ind, c in self.eq_elec_migration_flux_in.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.elec_migration_flux_in)
            )
        for ind, c in self.eq_elec_migration_flux_out.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.elec_migration_flux_out)
            )
        for ind, c in self.eq_nonelec_flux_in.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.nonelec_flux_in[ind])
            )

        for ind, c in self.eq_nonelec_flux_out.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.nonelec_flux_out[ind])
            )

        for ind, c in self.eq_mass_transfer_term_aem_side.items():
            iscale.constraint_scaling_transform(
                c,
                min(
                    iscale.get_scaling_factor(self.elec_migration_flux_in[ind]),
                    iscale.get_scaling_factor(
                        self.nonelec_flux_in[ind], self.elec_migration_flux_out[ind]
                    ),
                    iscale.get_scaling_factor(self.nonelec_flux_out[ind]),
                ),
            )
        for ind, c in self.eq_mass_transfer_term_cem_side.items():
            iscale.constraint_scaling_transform(
                c,
                min(
                    iscale.get_scaling_factor(self.elec_migration_flux_in[ind]),
                    iscale.get_scaling_factor(
                        self.nonelec_flux_in[ind], self.elec_migration_flux_out[ind]
                    ),
                    iscale.get_scaling_factor(self.nonelec_flux_out[ind]),
                ),
            )

        if hasattr(self, "eq_current_dens_lim_bpem"):
            for ind, c in self.eq_current_dens_lim_bpem.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.current_dens_lim_bpem[ind])
                )

        for ind, c in self.eq_recovery_mass_H2O.items():
            iscale.constraint_scaling_transform(
                c,
                iscale.get_scaling_factor(
                    self.aem_side.properties_out[ind].flow_mass_phase_comp["Liq", "H2O"]
                ),
            )

        for ind, c in self.eq_power_electrical.items():
            iscale.constraint_scaling_transform(
                c,
                iscale.get_scaling_factor(self.power_electrical[ind]),
            )

        for ind, c in self.eq_specific_power_electrical.items():
            iscale.constraint_scaling_transform(
                c,
                iscale.get_scaling_factor(self.specific_power_electrical[ind])
                * iscale.get_scaling_factor(
                    self.aem_side.properties_out[ind].flow_vol_phase["Liq"]
                ),
            )
        for ind, c in self.eq_current_efficiency.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.current[ind])
            )

    def _get_stream_table_contents(self, time_point=0):
        return create_stream_table_dataframe(
            {
                "aem side of the bipolar membrane Channel Inlet": self.inlet_aem_side,
                "cem side of the bipolar membrane Channel Inlet": self.inlet_cem_side,
                "aem side of the bipolar membrane Channel Outlet": self.outlet_aem_side,
                "cem side of the bipolar membrane Channel Outlet": self.outlet_cem_side,
            },
            time_point=time_point,
        )

    def _get_performance_contents(self, time_point=0):
        return {
            "vars": {
                "Total electrical power consumption(Watt)": self.power_electrical[
                    time_point
                ],
                "Specific electrical power consumption (kW*h/m**3)": self.specific_power_electrical[
                    time_point
                ],
                "Current efficiency for deionzation": self.current_efficiency[
                    time_point
                ],
                "Water recovery by mass": self.recovery_mass_H2O[time_point],
            },
            "exprs": {},
            "params": {},
        }

    def get_power_electrical(self, time_point=0):
        return self.power_electrical[time_point]

    @property
    def default_costing_method(self):
        return cost_electrodialysis
