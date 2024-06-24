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
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
import pytest
from idaes.core import (
    FlowsheetBlock,
)
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.testing import initialization_tester
from pyomo.environ import (
    ConcreteModel,
    assert_optimal_termination,
    value,
)

from electrodialysis_bmed import (
    BiPolarElectrodialysis0D,
    LimitingCurrentDensityMethod,
    LimitingpotentialMethod,
)
from watertap.core.solvers import get_solver
from watertap.property_models.multicomp_aq_sol_prop_pack import MCASParameterBlock

# from watertap.unit_models.electrodialysis_0D import LimitingCurrentDensityMethod

__author__ = "Johnson Dhanasekaran"

solver = get_solver()


# -----------------------------------------------------------------------------
# Start test class


class Test_operation_Method:
    @pytest.fixture(scope="class")
    def bped_sub_limiting(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-", "C_+", "A_-"],
            "mw_data": {
                "H2O": 18e-3,
                "Na_+": 23e-3,
                "Cl_-": 35.5e-3,
                "C_+": 1e-3,
                "A_-": 17.0e-3,
            },
            "elec_mobility_data": {
                ("Liq", "Na_+"): 5.19e-8,
                ("Liq", "Cl_-"): 7.92e-8,
                ("Liq", "C_+"): 36.23e-8,
                ("Liq", "A_-"): 20.64e-8,
            },
            "charge": {"Na_+": 1, "Cl_-": -1, "C_+": 1, "A_-": -1},
            "diffusivity_data": {
                ("Liq", "Na_+"): 1.33e-9,
                ("Liq", "Cl_-"): 2.03e-9,
                ("Liq", "C_+"): 9.31e-9,
                ("Liq", "A_-"): 5.27e-9,
            },
        }
        m.fs.properties = MCASParameterBlock(**ion_dict)
        m.fs.unit = BiPolarElectrodialysis0D(
            property_package=m.fs.properties,
            operation_mode="Constant_Current",
            Operation_method_bpem="Sub_limiting",
        )

        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "C_+"].fix(7.38e-4)
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "A_-"].fix(7.38e-4)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "C_+"].fix(7.38e-4)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "A_-"].fix(7.38e-4)

        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)

        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Na_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Cl_-")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "C_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "A_-")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e2, index=("Liq", "H2O")
        )
        m.fs.unit.cell_pair_num.fix(1)
        m.fs.unit.current.fix(8)
        return m

    @pytest.fixture(scope="class")
    def bped_over_limiting(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-", "C_+", "A_-"],
            "mw_data": {
                "H2O": 18e-3,
                "Na_+": 23e-3,
                "Cl_-": 35.5e-3,
                "C_+": 1e-3,
                "A_-": 17.0e-3,
            },
            "elec_mobility_data": {
                ("Liq", "Na_+"): 5.19e-8,
                ("Liq", "Cl_-"): 7.92e-8,
                ("Liq", "C_+"): 36.23e-8,
                ("Liq", "A_-"): 20.64e-8,
            },
            "charge": {"Na_+": 1, "Cl_-": -1, "C_+": 1, "A_-": -1},
            "diffusivity_data": {
                ("Liq", "Na_+"): 1.33e-9,
                ("Liq", "Cl_-"): 2.03e-9,
                ("Liq", "C_+"): 9.31e-9,
                ("Liq", "A_-"): 5.27e-9,
            },
        }
        m.fs.properties = MCASParameterBlock(**ion_dict)
        m.fs.unit = BiPolarElectrodialysis0D(
            property_package=m.fs.properties,
            operation_mode="Constant_Current",
            Operation_method_bpem="Over_limiting",
            limiting_current_density_method_bpem=LimitingCurrentDensityMethod.InitialValue,
            limiting_current_density_data=500,
            limiting_potential_method_bpem=LimitingpotentialMethod.InitialValue,
            limiting_potential_data=0.5,
        )

        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-1)
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-1)
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "C_+"].fix(7.38e-1)
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "A_-"].fix(7.38e-1)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-1)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-1)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "C_+"].fix(7.38e-1)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "A_-"].fix(7.38e-1)

        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e2)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e2)

        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e-1, index=("Liq", "H2O")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Na_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Cl_-")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "C_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "A_-")
        )

        m.fs.unit.cell_pair_num.fix(1)
        m.fs.unit.current.fix(1.5e3)
        return m

    @pytest.mark.unit
    def test_assign(self, bped_sub_limiting, bped_over_limiting):
        bped_m = (bped_sub_limiting, bped_over_limiting)
        # Specify a system
        # Due to lack of availability of data for bipolar membrane operation the CEM and AEM input fluxes have been
        # from Campione et al. in Desalination 465 (2019): 79-93.
        for m in bped_m:
            m.fs.unit.water_trans_number_membrane["bpem"].fix((5.8 + 4.3) / 2)
            m.fs.unit.water_permeability_membrane["bpem"].fix((2.16e-14 + 1.75e-14) / 2)
            m.fs.unit.electrodes_resistance.fix(0)
            m.fs.unit.current_utilization.fix(1)
            m.fs.unit.channel_height.fix(2.7e-4)
            m.fs.unit.membrane_areal_resistance["bpem"].fix((1.89e-4 + 1.77e-4) / 2)
            m.fs.unit.cell_width.fix(0.1)
            m.fs.unit.cell_length.fix(0.79)
            m.fs.unit.membrane_thickness["bpem"].fix(1.3e-4)
            m.fs.unit.solute_diffusivity_membrane["bpem", "Na_+"].fix(
                (1.8e-10 + 1.25e-10) / 2
            )
            m.fs.unit.solute_diffusivity_membrane["bpem", "Cl_-"].fix(
                (1.8e-10 + 1.25e-10) / 2
            )
            m.fs.unit.solute_diffusivity_membrane["bpem", "C_+"].fix(
                (1.8e-10 + 1.25e-10) / 2
            )
            m.fs.unit.solute_diffusivity_membrane["bpem", "A_-"].fix(
                (1.8e-10 + 1.25e-10) / 2
            )
            m.fs.unit.ion_trans_number_membrane["bpem", "Na_+"].fix(0.5)
            m.fs.unit.ion_trans_number_membrane["bpem", "Cl_-"].fix(0.5)
            m.fs.unit.ion_trans_number_membrane["bpem", "C_+"].fix(0.1)
            m.fs.unit.ion_trans_number_membrane["bpem", "A_-"].fix(0.1)
            m.fs.unit.solute_diffusivity_membrane["bpem", "Na_+"].fix(
                (1.8e-10 + 1.25e-10) / 2
            )
            m.fs.unit.solute_diffusivity_membrane["bpem", "Cl_-"].fix(
                (1.8e-10 + 1.25e-10) / 2
            )
            m.fs.unit.solute_diffusivity_membrane["bpem", "C_+"].fix(
                (1.8e-10 + 1.25e-10) / 2
            )
            m.fs.unit.solute_diffusivity_membrane["bpem", "A_-"].fix(
                (1.8e-10 + 1.25e-10) / 2
            )
            m.fs.unit.ion_trans_number_membrane["bpem", "Na_+"].fix(0.5)
            m.fs.unit.ion_trans_number_membrane["bpem", "Cl_-"].fix(0.5)
            m.fs.unit.ion_trans_number_membrane["bpem", "C_+"].fix(0.1)
            m.fs.unit.ion_trans_number_membrane["bpem", "A_-"].fix(0.1)

            # Set inlet stream. These have been scaled up by a factor of 1e3 from the ED process
            m.fs.unit.inlet_aem_side.pressure.fix(101325)
            m.fs.unit.inlet_aem_side.temperature.fix(298.15)
            m.fs.unit.inlet_cem_side.pressure.fix(101325)
            m.fs.unit.inlet_cem_side.temperature.fix(298.15)
            m.fs.unit.spacer_porosity.fix(1)

        # Since the comparison here is not against experimental data the critical aspects tested here are:
        # The zeroing of quantities and magnitude symmetries

        # Test sub limiting
        iscale.calculate_scaling_factors(bped_m[0].fs)
        assert degrees_of_freedom(bped_m[0]) == 0
        initialization_tester(bped_m[0], outlvl=idaeslog.DEBUG)
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(bped_m[0])
        }
        # assert not badly_scaled_var_values
        results = solver.solve(bped_m[0])
        assert_optimal_termination(results)
        assert value(
            bped_m[0].fs.unit.elec_migration_flux_out[0, "Liq", "Na_+"]
        ) == pytest.approx(0.0005248, rel=1e-2)
        assert value(
            bped_m[0].fs.unit.elec_migration_flux_out[0, "Liq", "Cl_-"]
        ) == pytest.approx(-0.0005248, rel=1e-2)
        assert value(
            bped_m[0].fs.unit.elec_migration_flux_out[0, "Liq", "C_+"]
        ) == pytest.approx(0, rel=1e-3)
        assert value(
            bped_m[0].fs.unit.elec_migration_flux_out[0, "Liq", "A_-"]
        ) == pytest.approx(0, rel=1e-3)
        assert value(
            bped_m[0].fs.unit.elec_migration_flux_out[0, "Liq", "H2O"]
        ) == pytest.approx(0.0053, rel=1e-3)

        # Test Over limiting
        iscale.calculate_scaling_factors(bped_m[1].fs)
        assert degrees_of_freedom(bped_m[1]) == 0
        initialization_tester(bped_m[1], outlvl=idaeslog.DEBUG)
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(bped_m[1])
        }
        # assert not badly_scaled_var_values
        results = solver.solve(bped_m[1])
        assert_optimal_termination(results)
        assert value(
            bped_m[1].fs.unit.elec_migration_flux_out[0, "Liq", "Na_+"]
        ) == pytest.approx(0.032798, rel=1e-3)
        assert value(
            bped_m[1].fs.unit.elec_migration_flux_out[0, "Liq", "Cl_-"]
        ) == pytest.approx(-0.032798, rel=1e-3)
        assert value(
            bped_m[1].fs.unit.elec_migration_flux_out[0, "Liq", "C_+"]
        ) == pytest.approx(0.1312, rel=1e-3)
        assert value(
            bped_m[1].fs.unit.elec_migration_flux_out[0, "Liq", "A_-"]
        ) == pytest.approx(-0.1312, rel=1e-3)
        assert value(
            bped_m[1].fs.unit.elec_migration_flux_out[0, "Liq", "H2O"]
        ) == pytest.approx(0.99389, rel=1e-3)


#
#
class Test_limiting_parameters:
    @pytest.fixture(scope="class")
    def limiting_current_check(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-", "C_+", "A_-"],
            "mw_data": {
                "H2O": 18e-3,
                "Na_+": 23e-3,
                "Cl_-": 35.5e-3,
                "C_+": 1e-3,
                "A_-": 17.0e-3,
            },
            "elec_mobility_data": {
                ("Liq", "Na_+"): 5.19e-8,
                ("Liq", "Cl_-"): 7.92e-8,
                ("Liq", "C_+"): 36.23e-8,
                ("Liq", "A_-"): 20.64e-8,
            },
            "charge": {"Na_+": 1, "Cl_-": -1, "C_+": 1, "A_-": -1},
            "diffusivity_data": {
                ("Liq", "Na_+"): 1.33e-9,
                ("Liq", "Cl_-"): 2.03e-9,
                ("Liq", "C_+"): 9.31e-9,
                ("Liq", "A_-"): 5.27e-9,
            },
        }
        m.fs.properties = MCASParameterBlock(**ion_dict)
        m.fs.unit = BiPolarElectrodialysis0D(
            property_package=m.fs.properties,
            operation_mode="Constant_Current",
            Operation_method_bpem="Over_limiting",
            limiting_current_density_method_bpem=LimitingCurrentDensityMethod.Empirical,
            limiting_potential_method_bpem=LimitingpotentialMethod.InitialValue,
            limiting_potential_data=0.5,
        )
        return m

    @pytest.fixture(scope="class")
    def potential_barrier_check(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-", "C_+", "A_-"],
            "mw_data": {
                "H2O": 18e-3,
                "Na_+": 23e-3,
                "Cl_-": 35.5e-3,
                "C_+": 1e-3,
                "A_-": 17.0e-3,
            },
            "elec_mobility_data": {
                ("Liq", "Na_+"): 5.19e-8,
                ("Liq", "Cl_-"): 7.92e-8,
                ("Liq", "C_+"): 36.23e-8,
                ("Liq", "A_-"): 20.64e-8,
            },
            "charge": {"Na_+": 1, "Cl_-": -1, "C_+": 1, "A_-": -1},
            "diffusivity_data": {
                ("Liq", "Na_+"): 1.33e-9,
                ("Liq", "Cl_-"): 2.03e-9,
                ("Liq", "C_+"): 9.31e-9,
                ("Liq", "A_-"): 5.27e-9,
            },
        }
        m.fs.properties = MCASParameterBlock(**ion_dict)
        m.fs.unit = BiPolarElectrodialysis0D(
            property_package=m.fs.properties,
            operation_mode="Constant_Current",
            Operation_method_bpem="Over_limiting",
            limiting_current_density_method_bpem=LimitingCurrentDensityMethod.Empirical,
            limiting_potential_method_bpem=LimitingpotentialMethod.Empirical,
        )

        m.fs.unit.kr["bpem"].fix(1.33 * 10**11)
        m.fs.unit.kd_zero["bpem"].fix(2 * 10**-5)
        m.fs.unit.relative_permittivity["bpem"].fix(20)

        return m

    @pytest.mark.unit
    def test_assign(self, limiting_current_check, potential_barrier_check):
        check_m = (limiting_current_check, potential_barrier_check)
        for m in check_m:
            m.fs.unit.current.fix(1.5e3)
            m.fs.unit.water_trans_number_membrane["bpem"].fix((5.8 + 4.3) / 2)
            m.fs.unit.water_permeability_membrane["bpem"].fix((2.16e-14 + 1.75e-14) / 2)
            m.fs.unit.electrodes_resistance.fix(0)
            m.fs.unit.cell_pair_num.fix(1)
            m.fs.unit.current_utilization.fix(1)
            m.fs.unit.channel_height.fix(2.7e-4)
            m.fs.unit.membrane_areal_resistance["bpem"].fix((1.89e-4 + 1.77e-4) / 2)
            m.fs.unit.cell_width.fix(0.1)
            m.fs.unit.cell_length.fix(0.79)
            m.fs.unit.membrane_thickness["bpem"].fix(1.3e-4)
            m.fs.unit.diffus_mass["bpem"].fix((2.03 + 1.96) * 10**-9 / 2)
            m.fs.unit.solute_diffusivity_membrane["bpem", "Na_+"].fix(
                (1.8e-10 + 1.25e-10) / 2
            )
            m.fs.unit.solute_diffusivity_membrane["bpem", "Cl_-"].fix(
                (1.8e-10 + 1.25e-10) / 2
            )
            m.fs.unit.solute_diffusivity_membrane["bpem", "C_+"].fix(
                (1.8e-10 + 1.25e-10) / 2
            )
            m.fs.unit.solute_diffusivity_membrane["bpem", "A_-"].fix(
                (1.8e-10 + 1.25e-10) / 2
            )
            m.fs.unit.ion_trans_number_membrane["bpem", "Na_+"].fix(0.5)
            m.fs.unit.ion_trans_number_membrane["bpem", "Cl_-"].fix(0.5)
            m.fs.unit.ion_trans_number_membrane["bpem", "C_+"].fix(0.1)
            m.fs.unit.ion_trans_number_membrane["bpem", "A_-"].fix(0.1)
            m.fs.unit.solute_diffusivity_membrane["bpem", "Na_+"].fix(
                (1.8e-10 + 1.25e-10) / 2
            )
            m.fs.unit.solute_diffusivity_membrane["bpem", "Cl_-"].fix(
                (1.8e-10 + 1.25e-10) / 2
            )
            m.fs.unit.solute_diffusivity_membrane["bpem", "C_+"].fix(
                (1.8e-10 + 1.25e-10) / 2
            )
            m.fs.unit.solute_diffusivity_membrane["bpem", "A_-"].fix(
                (1.8e-10 + 1.25e-10) / 2
            )
            m.fs.unit.ion_trans_number_membrane["bpem", "Na_+"].fix(0.5)
            m.fs.unit.ion_trans_number_membrane["bpem", "Cl_-"].fix(0.5)
            m.fs.unit.ion_trans_number_membrane["bpem", "C_+"].fix(0.1)
            m.fs.unit.ion_trans_number_membrane["bpem", "A_-"].fix(0.1)
            m.fs.unit.kr["bpem"].fix(1.33 * 10**11)
            m.fs.unit.kd_zero["bpem"].fix(2 * 10**-5)
            m.fs.unit.relative_permittivity["bpem"].fix(20)
            m.fs.unit.diffus_mass["bpem"].fix((2.03 + 1.96) * 10**-9 / 2)
            m.fs.unit.salt_conc_aem["bpem"].fix(500)
            m.fs.unit.salt_conc_cem["bpem"].fix(500)
            m.fs.unit.membrane_fixed_charge["bpem"].fix(1.5e3)

            m.fs.unit.inlet_aem_side.pressure.fix(101325)
            m.fs.unit.inlet_aem_side.temperature.fix(298.15)
            m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
            m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-1)
            m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-1)
            m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "C_+"].fix(7.38e-1)
            m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "A_-"].fix(7.38e-1)
            m.fs.unit.inlet_cem_side.pressure.fix(101325)
            m.fs.unit.inlet_cem_side.temperature.fix(298.15)
            m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
            m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-1)
            m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-1)
            m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "C_+"].fix(7.38e-1)
            m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "A_-"].fix(7.38e-1)
            m.fs.unit.spacer_porosity.fix(1)
            m.fs.properties.set_default_scaling(
                "flow_mol_phase_comp", 1e1, index=("Liq", "H2O")
            )
            m.fs.properties.set_default_scaling(
                "flow_mol_phase_comp", 1e0, index=("Liq", "Na_+")
            )
            m.fs.properties.set_default_scaling(
                "flow_mol_phase_comp", 1e0, index=("Liq", "Cl_-")
            )
            m.fs.properties.set_default_scaling(
                "flow_mol_phase_comp", 1e0, index=("Liq", "C_+")
            )
            m.fs.properties.set_default_scaling(
                "flow_mol_phase_comp", 1e0, index=("Liq", "A_-")
            )

        # Data on limiting current and potential barrier to water splitting have been obtained from:
        # Fumatech, Technical Data Sheet for Fumasep FBM, 2020. With additional modelling parameters obtaiend from
        # Ionescu, Viorel. Advanced Topics in Optoelectronics, Microelectronics, and Nanotechnologies (2023)
        iscale.calculate_scaling_factors(check_m[1])

        # Test computing limiting current in  bipolar membrane
        iscale.calculate_scaling_factors(check_m[0])
        assert degrees_of_freedom(check_m[0]) == 0
        initialization_tester(check_m[0], outlvl=idaeslog.DEBUG)
        badly_scaled_var_values = {
            var.name: val
            for (var, val) in iscale.badly_scaled_var_generator(check_m[0])
        }
        assert not badly_scaled_var_values
        results = solver.solve(check_m[0])
        assert_optimal_termination(results)
        assert value(check_m[0].fs.unit.current_dens_lim_bpem[0]) == pytest.approx(
            987, rel=1e-3
        )

        # Test computing limiting current and potential barrier to water splitting in  bipolar membrane
        iscale.calculate_scaling_factors(check_m[1])
        assert degrees_of_freedom(check_m[1]) == 0
        initialization_tester(check_m[1], outlvl=idaeslog.DEBUG)
        badly_scaled_var_values = {
            var.name: val
            for (var, val) in iscale.badly_scaled_var_generator(check_m[1])
        }
        assert not badly_scaled_var_values
        results = solver.solve(check_m[1])
        assert_optimal_termination(results)
        assert value(check_m[1].fs.unit.current_dens_lim_bpem[0]) == pytest.approx(
            987, rel=1e-3
        )
        assert value(check_m[1].fs.unit.potential_lim_bpem[0]) == pytest.approx(
            0.786, rel=1e-3
        )
