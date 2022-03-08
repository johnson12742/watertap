###############################################################################
# WaterTAP Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################

from enum import Enum

import pyomo.environ as pyo

from pyomo.util.calc_var_value import calculate_variable_from_constraint

from idaes.core.util.exceptions import BurntToast, ConfigurationError
from idaes.core.util.misc import StrEnum
from idaes.core import declare_process_block_class
from idaes.generic_models.costing.costing_base import (
    FlowsheetCostingBlockData, register_idaes_currency_units)

from watertap.unit_models import (
        ReverseOsmosis0D,
        ReverseOsmosis1D,
        NanoFiltration0D,
        NanoFiltrationZO,
        PressureExchanger,
        Pump,
        )


class ROType(StrEnum):
    standard = "standard"
    high_pressure = "high_pressure"


class PumpType(StrEnum):
    low_pressure = "low_pressure"
    high_pressure = "high_pressure"
    pressure_exchanger = "pressure_exchanger"


@declare_process_block_class("WaterTAPCosting")
class WaterTAPCostingData(FlowsheetCostingBlockData):

    def build_global_params(self):

        # Register currency and conversion rates based on CE Index
        register_idaes_currency_units()

        # Set the base year for all costs
        self.base_currency = pyo.units.USD_CE500
        # Set a base period for all operating costs
        self.base_period = pyo.units.year

        # Define standard material flows and costs
        self.defined_flows["electricity"] = 0.07 * pyo.units.USD_CE500 / pyo.units.kWh

        # Build flowsheet level costing components
        # This is package specific
        self.load_factor = pyo.Var(
                initialize=0.9,
                doc='Load factor [fraction of uptime]',
                units=pyo.units.dimensionless)
        self.factor_total_investment = pyo.Var(
                initialize=2,
                doc='Total investment factor [investment cost/equipment cost]',
                units=pyo.units.dimensionless)
        self.factor_maintenance_labor_chemical = pyo.Var(
                initialize=0.03,
                doc='Maintenance-labor-chemical factor [fraction of investment cost/year]',
                units=pyo.units.year**-1)
        self.factor_capital_annualization = pyo.Var(
                initialize=0.1,
                doc='Capital annualization factor [fraction of investment cost/year]',
                units=pyo.units.year**-1)
        self.factor_membrane_replacement = pyo.Var(
                initialize=0.2,
                doc='Membrane replacement factor [fraction of membrane replaced/year]',
                units=pyo.units.year**-1)
        self.reverse_osmosis_membrane_cost = pyo.Var(
                initialize=30,
                doc='Membrane cost',
                units=pyo.units.USD_CE500/(pyo.units.meter**2))
        self.reverse_osmosis_high_pressure_membrane_cost = pyo.Var(
                initialize=75,
                doc='Membrane cost',
                units=pyo.units.USD_CE500/(pyo.units.meter**2))
        self.nanofiltration_membrane_cost = pyo.Var(
                initialize=15,
                doc='Membrane cost',
                units=pyo.units.USD_CE500/(pyo.units.meter**2))
        self.high_pressure_pump_cost = pyo.Var(
                initialize=53 / 1e5 * 3600,
                doc='High pressure pump cost',
                units=pyo.units.USD_CE500/pyo.units.watt)
        self.low_pressure_pump_cost = pyo.Var(
                initialize=889,
                doc='Low pressure pump cost',
                units=pyo.units.USD_CE500/(pyo.units.liter/pyo.units.second))
        self.pump_pressure_exchanger_cost = pyo.Var(
                initialize=535,
                doc='Pressure exchanger cost',
                units=pyo.units.USD_CE500/(pyo.units.meter**3/pyo.units.hours))
        self.pressure_exchanger_cost = pyo.Var(
                initialize=535,
                doc='Pressure exchanger cost',
                units=pyo.units.USD_CE500/(pyo.units.meter**3/pyo.units.hours))

        # fix the parameters
        for var in self.component_objects(pyo.Var):
            var.fix()

    def build_process_costs(self):
        self.total_capital_cost = pyo.Expression(
                expr = self.aggregate_capital_cost,
                doc='Total capital cost [$]')
        self.total_investment_cost = pyo.Var(
                initialize=1e3,
                domain=pyo.NonNegativeReals,
                doc='Total investment cost [$]',
                units=pyo.units.USD_CE500)
        self.maintenance_labor_chemical_operating_cost = pyo.Var(
                initialize=1e3,
                domain=pyo.NonNegativeReals,
                doc='Maintenance-labor-chemical operating cost [$/year]',
                units=pyo.units.USD_CE500/pyo.units.year)
        self.total_operating_cost= pyo.Var(
                initialize=1e3,
                domain=pyo.NonNegativeReals,
                doc='Total operating cost [$/year]',
                units=pyo.units.USD_CE500/pyo.units.year)

        self.total_investment_cost_constraint = pyo.Constraint(expr = \
                self.total_investment_cost == self.factor_total_investment * self.total_capital_cost)
        self.maintenance_labor_chemical_operating_cost_constraint = pyo.Constraint(expr = \
                self.maintenance_labor_chemical_operating_cost == self.factor_maintenance_labor_chemical * self.total_investment_cost)

        self.total_operating_cost_constraint = pyo.Constraint(expr = \
                self.total_operating_cost == self.maintenance_labor_chemical_operating_cost \
                + self.aggregate_fixed_operating_cost \
                + self.aggregate_variable_operating_cost \
                + sum(self.aggregate_flow_costs.values())*self.load_factor )

    def initialize_build(self):
        calculate_variable_from_constraint(self.total_investment_cost, self.total_investment_cost_constraint)
        calculate_variable_from_constraint(self.maintenance_labor_chemical_operating_cost,
                self.maintenance_labor_chemical_operating_cost_constraint)
        calculate_variable_from_constraint(self.total_operating_cost,
                self.total_operating_cost_constraint)

        if hasattr(self, "LCOW"):
            calculate_variable_from_constraint(self.LCOW, self.LCOW_constraint)

    def add_LCOW(self, flow_rate):
        """
        Add Levelized Cost of Water (LCOW) to costing block.
        Args:
            flow_rate - flow rate of water (volumetric) to be used in
                        calculating LCOW
        """

        self.annual_water_production = pyo.Expression( expr=
                 (pyo.units.convert(
                     flow_rate,
                     to_units=pyo.units.m**3/self.base_period) *
                  self.load_factor))

        self.LCOW = pyo.Var(
            doc='Levelized Cost of Water',
            units=self.base_currency/pyo.units.m**3)

        self.LCOW_constraint = pyo.Constraint(expr = self.LCOW == 
            (self.total_investment_cost*self.factor_capital_annualization +
                  self.total_operating_cost) / self.annual_water_production)

    def add_specific_energy_consumption(self, flow_rate):
        """
        Add specific energy consumption (kWh/m**3) to costing block.
        Args:
            flow_rate - flow rate of water (volumetric) to be used in
                        calculating specific energy consumption
        """

        self.specific_energy_consumption = pyo.Expression(expr = 
                self.aggregate_flow_electricity / pyo.units.convert(flow_rate, to_units=pyo.units.m**3/pyo.units.hr))

    # Define costing methods supported by package
    @staticmethod
    def cost_nanofiltration(blk):
        """
        Nanofiltration costing method

        TODO: describe equations
        """
        _make_captial_cost_var(blk)
        _make_fixed_operating_cost_var(blk)

        cost_membrane(blk, blk.costing_package.nanofiltration_membrane_cost, blk.costing_package.factor_membrane_replacement)

    @staticmethod
    def cost_reverse_osmosis(blk, ro_type=ROType.standard):
        """
        Reverse osmosis costing method

        TODO: describe equations

        Args:
            ro_type - ROType Enum indicating reverse osmosis type,
                      default = ROType.standard
        """
        # Validate arguments
        if ro_type not in ROType:
            raise ConfigurationError(
                f"{blk.unit_model.name} received invalid argument for ro_type:"
                f" {ro_type}. Argument must be a member of the ROType Enum.")

        _make_captial_cost_var(blk)
        _make_fixed_operating_cost_var(blk)

        if ro_type == ROType.standard:
            membrane_cost = blk.costing_package.reverse_osmosis_membrane_cost
        elif ro_type == ROType.high_pressure:
            membrane_cost = blk.costing_package.reverse_osmosis_high_pressure_membrane_cost
        else:
            raise BurntToast(f"Unrecognized ro_type: {ro_type}")
        cost_membrane(blk, membrane_cost, blk.costing_package.factor_membrane_replacement)

    @staticmethod
    def cost_pump(blk, pump_type=PumpType.high_pressure):
        """
        Pump costing method

        TODO: describe equations

        Args:
            pump_type - PumpType Enum indicating pump type,
                        default = PumpType.high_pressure
        """
        if pump_type not in PumpType:
            raise ConfigurationError(
                f"{blk.unit_model.name} received invalid argument for pump_type:"
                f" {pump_type}. Argument must be a member of the PumpType Enum.")

        if pump_type == PumpType.high_pressure:
            WaterTAPCostingData.cost_high_pressure_pump(blk)
        elif pump_type == PumpType.low_pressure:
            WaterTAPCostingData.cost_low_pressure_pump(blk)
        elif pump_type == PumpType.pressure_exchanger:
            WaterTAPCostingData.cost_pressure_exchanger_pump(blk)
        else:
            raise BurntToast(f"Unrecognized pump_type: {pump_type}")

    @staticmethod
    def cost_high_pressure_pump(blk):
        """
        High pressure pump costing method

        TODO: describe equations
        """
        _make_captial_cost_var(blk)
        blk.capital_cost_constraint = pyo.Constraint(expr = \
                blk.capital_cost == blk.costing_package.high_pressure_pump_cost * pyo.units.convert(blk.unit_model.work_mechanical[0], pyo.units.W))

    @staticmethod
    def cost_low_pressure_pump(blk):
        """
        High pressure pump costing method

        TODO: describe equations
        """
        _make_captial_cost_var(blk)
        cost_by_flow_volume(blk, blk.costing_package.low_pressure_pump_cost,
                pyo.units.convert(blk.unit_model.control_volume.properties_in[0].flow_vol, (pyo.units.m**3/pyo.units.s)))

    @staticmethod
    def cost_pressure_exchanger_pump(blk):
        """
        Pump pressure exchanger costing method

        TODO: describe equations
        """
        _make_captial_cost_var(blk)
        cost_by_flow_volume(blk, blk.costing_package.pump_pressure_exchanger_cost,
                pyo.units.convert(blk.unit_model.control_volume.properties_in[0].flow_vol, (pyo.units.meter**3/pyo.units.hours)))

    @staticmethod
    def cost_pressure_exchanger(blk):
        """
        Pressure exchanger costing method

        TODO: describe equations
        """
        _make_captial_cost_var(blk)
        cost_by_flow_volume(blk, blk.costing_package.pressure_exchanger_cost,
                pyo.units.convert(blk.unit_model.low_pressure_side.properties_in[0].flow_vol, (pyo.units.meter**3/pyo.units.hours)))

    ## TODO; Mixer and Separator

# Define default mapping of costing methods to unit models
WaterTAPCostingData.unit_mapping = {
        Pump: WaterTAPCostingData.cost_pump,
        PressureExchanger: WaterTAPCostingData.cost_pressure_exchanger,
        ReverseOsmosis0D: WaterTAPCostingData.cost_reverse_osmosis,
        ReverseOsmosis1D: WaterTAPCostingData.cost_reverse_osmosis,
        NanoFiltration0D: WaterTAPCostingData.cost_nanofiltration,
        NanoFiltrationZO: WaterTAPCostingData.cost_nanofiltration,
        }

def _make_captial_cost_var(blk):
    blk.capital_cost = pyo.Var(initialize=1e5,
                           domain=pyo.NonNegativeReals,
                           units=pyo.units.USD_CE500,
                           doc="Unit capital cost")

def _make_fixed_operating_cost_var(blk):
    blk.fixed_operating_cost = pyo.Var(initialize=1e5,
                                   domain=pyo.NonNegativeReals,
                                   units=pyo.units.USD_CE500/pyo.units.year,
                                   doc="Unit fixed operating cost")

def cost_membrane(blk, membrane_cost, factor_membrane_replacement):
    """
    Generic function for costing a membrane. Assumes the unit_model
    has an `area` variable or parameter.

    Args:
        membrane_cost - The cost of the membrane in currency per area
        factor_membrane_replacement - Membrane replacement factor
                                      [fraction of membrane replaced/year]

    """
    blk.membrane_cost = pyo.Expression(expr=membrane_cost)
    blk.factor_membrane_replacement = pyo.Expression(expr=factor_membrane_replacement)

    blk.capital_cost_constraint = pyo.Constraint(expr = \
            blk.capital_cost == blk.membrane_cost * pyo.units.convert(blk.unit_model.area, pyo.units.m**2))
    blk.fixed_operating_cost_constraint = pyo.Constraint(expr = \
            blk.fixed_operating_cost == blk.factor_membrane_replacement * blk.membrane_cost * pyo.units.convert(blk.unit_model.area, pyo.units.m**2))

def cost_by_flow_volume(blk, flow_cost, flow_to_cost):
    """
    Generic function for costing by flow volume.

    Args:
        flow_cost - The cost of the pump in [currency]/([volume]/[time])
        flow_to_cost - The flow costed in [volume]/[time]
    """
    blk.flow_cost = pyo.Expression(expr=flow_cost)
    blk.capital_cost_constraint = pyo.Constraint(expr = \
            blk.capital_cost == blk.flow_cost * flow_to_cost)
