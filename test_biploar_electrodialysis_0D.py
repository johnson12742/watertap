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
from idaes.core.util.constants import Constants
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

from electrodialysis_bipolar import (
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

    @pytest.mark.unit
    def test_oepration(self):
        bped_current = [5, 10, 50, 100]
        # Specify a system
        # Due to lack of availability of data for bipolar membrane operation the CEM and AEM input fluxes have been
        # from Campione et al. in Desalination 465 (2019): 79-93.

        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-", "H_+", "OH_-"],
            "mw_data": {
                "H2O": 18e-3,
                "Na_+": 23e-3,
                "Cl_-": 35.5e-3,
                "H_+": 1e-3,
                "OH_-": 17.0e-3,
            },
            "elec_mobility_data": {
                ("Liq", "Na_+"): 5.19e-8,
                ("Liq", "Cl_-"): 7.92e-8,
                ("Liq", "H_+"): 36.23e-8,
                ("Liq", "OH_-"): 20.64e-8,
            },
            "charge": {"Na_+": 1, "Cl_-": -1, "H_+": 1, "OH_-": -1},
            "diffusivity_data": {
                ("Liq", "Na_+"): 1.33e-9,
                ("Liq", "Cl_-"): 2.03e-9,
                ("Liq", "H_+"): 9.31e-9,
                ("Liq", "OH_-"): 5.27e-9,
            },
        }
        m.fs.properties = MCASParameterBlock(**ion_dict)
        m.fs.unit = BiPolarElectrodialysis0D(
            property_package=m.fs.properties,
            has_catalyst=False,
            limiting_current_density_method_bpem=LimitingCurrentDensityMethod.InitialValue,
            limiting_current_density_data=500,
            limiting_potential_method_bpem=LimitingpotentialMethod.InitialValue,
            limiting_potential_data=0.5,
        )

        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "H_+"].fix(7.38e-4)
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "OH_-"].fix(7.38e-4)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "H_+"].fix(7.38e-4)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "OH_-"].fix(7.38e-4)

        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)

        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Na_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Cl_-")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "H_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "OH_-")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "H2O")
        )

        m.fs.unit.water_trans_number_membrane["bpem"].fix((5.8 + 4.3) / 2)
        m.fs.unit.water_permeability_membrane["bpem"].fix((2.16e-14 + 1.75e-14) / 2)
        m.fs.unit.electrodes_resistance.fix(0)
        m.fs.unit.cell_pair_num.fix(2)
        m.fs.unit.current_utilization.fix(1)
        m.fs.unit.channel_height.fix(2.7e-4)
        m.fs.unit.membrane_areal_resistance["bpem"].fix((1.89e-4 + 1.77e-4) / 2)
        m.fs.unit.cell_width.fix(0.1)
        m.fs.unit.cell_length.fix(0.79)
        m.fs.unit.membrane_thickness["bpem"].fix(1.3e-4)
        m.fs.unit.ion_trans_number_membrane["bpem", "Na_+"].fix(0.5)
        m.fs.unit.ion_trans_number_membrane["bpem", "Cl_-"].fix(0.5)
        m.fs.unit.ion_trans_number_membrane["bpem", "H_+"].fix(0.1)
        m.fs.unit.ion_trans_number_membrane["bpem", "OH_-"].fix(0.1)

        # Set inlet stream. These have been scaled up by a factor of 1e3 from the ED process
        m.fs.unit.inlet_aem_side.pressure.fix(101325)
        m.fs.unit.inlet_aem_side.temperature.fix(298.15)
        m.fs.unit.inlet_cem_side.pressure.fix(101325)
        m.fs.unit.inlet_cem_side.temperature.fix(298.15)
        m.fs.unit.spacer_porosity.fix(1)

        for i in range(len(bped_current)):
            m.fs.unit.current.fix(bped_current[i])

            # Since the comparison here is not against experimental data the critical aspects tested here are:
            # The zeroing of quantities and magnitude symmetries

            # Test sub limiting
            if i == 0:
                iscale.calculate_scaling_factors(m.fs)
                assert degrees_of_freedom(m) == 0
                initialization_tester(m, outlvl=idaeslog.DEBUG)
                badly_scaled_var_values = {
                    var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
                }
                # assert not badly_scaled_var_values
                results = solver.solve(m)
                assert_optimal_termination(results)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "Na_+"]
                ) == pytest.approx(-5.182e-5, rel=1e-3)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "Cl_-"]
                ) == pytest.approx(5.182e-5, rel=1e-3)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "H_+"]
                ) == pytest.approx(0, rel=1e-3)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "OH_-"]
                ) == pytest.approx(0, rel=1e-3)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "H2O"]
                ) == pytest.approx(-0.000523, rel=1e-3)

            if i == 1:
                iscale.calculate_scaling_factors(m.fs)
                assert degrees_of_freedom(m) == 0
                initialization_tester(m, outlvl=idaeslog.DEBUG)
                badly_scaled_var_values = {
                    var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
                }
                # assert not badly_scaled_var_values
                results = solver.solve(m)
                assert_optimal_termination(results)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "Na_+"]
                ) == pytest.approx(-0.0001036, rel=1e-3)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "Cl_-"]
                ) == pytest.approx(0.0001036, rel=1e-3)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "H_+"]
                ) == pytest.approx(0, rel=1e-3)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "OH_-"]
                ) == pytest.approx(0, rel=1e-3)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "H2O"]
                ) == pytest.approx(-0.001047, rel=1e-3)

            # Test over limiting
            if i == 2:
                iscale.calculate_scaling_factors(m.fs)
                assert degrees_of_freedom(m) == 0
                initialization_tester(m, outlvl=idaeslog.DEBUG)
                badly_scaled_var_values = {
                    var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
                }
                assert not badly_scaled_var_values
                results = solver.solve(m)
                assert_optimal_termination(results)

                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "Na_+"]
                ) == pytest.approx(-0.000409, rel=1e-3)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "Cl_-"]
                ) == pytest.approx(0.000409, rel=1e-3)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "H_+"]
                ) == pytest.approx(0, rel=1e-3)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "OH_-"]
                ) == pytest.approx(0.0002176, rel=1e-3)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "H2O"]
                ) == pytest.approx(-0.00523, rel=1e-3)
                assert value(
                    m.fs.unit.cem_side.mass_transfer_term[0, "Liq", "H_+"]
                ) == pytest.approx(0.0002176, rel=1e-3)
                assert value(
                    m.fs.unit.cem_side.mass_transfer_term[0, "Liq", "OH_-"]
                ) == pytest.approx(0, rel=1e-3)

            if i == 3:
                iscale.calculate_scaling_factors(m.fs)
                assert degrees_of_freedom(m) == 0
                initialization_tester(m, outlvl=idaeslog.DEBUG)
                badly_scaled_var_values = {
                    var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
                }
                assert not badly_scaled_var_values
                results = solver.solve(m)
                assert_optimal_termination(results)

                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "Na_+"]
                ) == pytest.approx(-0.000409, rel=1e-3)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "Cl_-"]
                ) == pytest.approx(0.000409, rel=1e-3)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "H_+"]
                ) == pytest.approx(0, rel=1e-3)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "OH_-"]
                ) == pytest.approx(0.001254, rel=1e-3)
                assert value(
                    m.fs.unit.aem_side.mass_transfer_term[0, "Liq", "H2O"]
                ) == pytest.approx(-0.01047, rel=1e-3)
                assert value(
                    m.fs.unit.cem_side.mass_transfer_term[0, "Liq", "H_+"]
                ) == pytest.approx(0.001254, rel=1e-3)
                assert value(
                    m.fs.unit.cem_side.mass_transfer_term[0, "Liq", "OH_-"]
                ) == pytest.approx(0, rel=1e-3)


class Test_limiting_parameters:
    @pytest.fixture(scope="class")
    def limiting_current_check(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-", "H_+", "OH_-"],
            "mw_data": {
                "H2O": 18e-3,
                "Na_+": 23e-3,
                "Cl_-": 35.5e-3,
                "H_+": 1e-3,
                "OH_-": 17.0e-3,
            },
            "elec_mobility_data": {
                ("Liq", "Na_+"): 5.19e-8,
                ("Liq", "Cl_-"): 7.92e-8,
                ("Liq", "H_+"): 36.23e-8,
                ("Liq", "OH_-"): 20.64e-8,
            },
            "charge": {"Na_+": 1, "Cl_-": -1, "H_+": 1, "OH_-": -1},
            "diffusivity_data": {
                ("Liq", "Na_+"): 1.33e-9,
                ("Liq", "Cl_-"): 2.03e-9,
                ("Liq", "H_+"): 9.31e-9,
                ("Liq", "OH_-"): 5.27e-9,
            },
        }
        m.fs.properties = MCASParameterBlock(**ion_dict)
        m.fs.unit = BiPolarElectrodialysis0D(
            property_package=m.fs.properties,
            has_catalyst=False,
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
            "solute_list": ["Na_+", "Cl_-", "H_+", "OH_-"],
            "mw_data": {
                "H2O": 18e-3,
                "Na_+": 23e-3,
                "Cl_-": 35.5e-3,
                "H_+": 1e-3,
                "OH_-": 17.0e-3,
            },
            "elec_mobility_data": {
                ("Liq", "Na_+"): 5.19e-8,
                ("Liq", "Cl_-"): 7.92e-8,
                ("Liq", "H_+"): 36.23e-8,
                ("Liq", "OH_-"): 20.64e-8,
            },
            "charge": {"Na_+": 1, "Cl_-": -1, "H_+": 1, "OH_-": -1},
            "diffusivity_data": {
                ("Liq", "Na_+"): 1.33e-9,
                ("Liq", "Cl_-"): 2.03e-9,
                ("Liq", "H_+"): 9.31e-9,
                ("Liq", "OH_-"): 5.27e-9,
            },
        }
        m.fs.properties = MCASParameterBlock(**ion_dict)
        m.fs.unit = BiPolarElectrodialysis0D(
            property_package=m.fs.properties,
            has_catalyst=False,
            limiting_current_density_method_bpem=LimitingCurrentDensityMethod.Empirical,
            limiting_potential_method_bpem=LimitingpotentialMethod.Empirical,
        )
        return m

    @pytest.mark.unit
    def test_assign(self, limiting_current_check, potential_barrier_check):
        check_m = (limiting_current_check, potential_barrier_check)
        for m in check_m:
            m.fs.unit.current.fix(5e2)
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
            m.fs.unit.ion_trans_number_membrane["bpem", "Na_+"].fix(0.5)
            m.fs.unit.ion_trans_number_membrane["bpem", "Cl_-"].fix(0.5)
            m.fs.unit.ion_trans_number_membrane["bpem", "H_+"].fix(0.1)
            m.fs.unit.ion_trans_number_membrane["bpem", "OH_-"].fix(0.1)
            m.fs.unit.conc_water["bpem"].fix(55 * 1e3)
            m.fs.unit.kr["bpem"].fix(1.33 * 10**11)
            m.fs.unit.k2_zero["bpem"].fix(2 * 10**-5)
            m.fs.unit.relative_permittivity["bpem"].fix(20)
            m.fs.unit.diffus_mass["bpem"].fix((2.03 + 1.96) * 10**-9 / 2)
            m.fs.unit.salt_conc_aem["bpem"].fix(500 + 2 * 250)
            m.fs.unit.salt_conc_cem["bpem"].fix(500 + 2 * 250)
            m.fs.unit.membrane_fixed_charge["bpem"].fix(1.5e3)

            m.fs.unit.inlet_aem_side.pressure.fix(101325)
            m.fs.unit.inlet_aem_side.temperature.fix(298.15)
            m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
            m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
            m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)
            m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "H_+"].fix(7.38e-4)
            m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "OH_-"].fix(7.38e-4)
            m.fs.unit.inlet_cem_side.pressure.fix(101325)
            m.fs.unit.inlet_cem_side.temperature.fix(298.15)
            m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
            m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
            m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)
            m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "H_+"].fix(7.38e-4)
            m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "OH_-"].fix(7.38e-4)
            m.fs.unit.spacer_porosity.fix(1)
            m.fs.properties.set_default_scaling(
                "flow_mol_phase_comp", 1e3, index=("Liq", "H2O")
            )
            m.fs.properties.set_default_scaling(
                "flow_mol_phase_comp", 1e3, index=("Liq", "Na_+")
            )
            m.fs.properties.set_default_scaling(
                "flow_mol_phase_comp", 1e3, index=("Liq", "Cl_-")
            )
            m.fs.properties.set_default_scaling(
                "flow_mol_phase_comp", 1e3, index=("Liq", "H_+")
            )
            m.fs.properties.set_default_scaling(
                "flow_mol_phase_comp", 1e3, index=("Liq", "OH_-")
            )

        # Data on limiting current and potential barrier to water splitting have been obtained from:
        # Fumatech, Technical Data Sheet for Fumasep FBM, 2020. With additional modelling parameters obtaiend from
        # Ionescu, Viorel. Advanced Topics in Optoelectronics, Microelectronics, and Nanotechnologies (2023)
        iscale.calculate_scaling_factors(check_m[0])

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
            1000, rel=1e-1
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
            1000, rel=1e-1
        )
        assert value(check_m[1].fs.unit.potential_lim_bpem[0]) == pytest.approx(
            0.8, rel=1e-1
        )


class Test_catalyst:

    @pytest.mark.unit
    def test_catalyst_current_generation(self):
        # Voltage applied across the bipolar membrane
        membrane_voltage = [0.6, 0.8, 1, 1.2]
        # Specify a system

        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-", "H_+", "OH_-"],
            "mw_data": {
                "H2O": 18e-3,
                "Na_+": 23e-3,
                "Cl_-": 35.5e-3,
                "H_+": 1e-3,
                "OH_-": 17.0e-3,
            },
            "elec_mobility_data": {
                ("Liq", "Na_+"): 5.19e-8,
                ("Liq", "Cl_-"): 7.92e-8,
                ("Liq", "H_+"): 36.23e-8,
                ("Liq", "OH_-"): 20.64e-8,
            },
            "charge": {"Na_+": 1, "Cl_-": -1, "H_+": 1, "OH_-": -1},
            "diffusivity_data": {
                ("Liq", "Na_+"): 1.33e-9,
                ("Liq", "Cl_-"): 2.03e-9,
                ("Liq", "H_+"): 9.31e-9,
                ("Liq", "OH_-"): 5.27e-9,
            },
        }
        m.fs.properties = MCASParameterBlock(**ion_dict)
        m.fs.unit = BiPolarElectrodialysis0D(
            property_package=m.fs.properties,
            has_catalyst=True,
            limiting_current_density_method_bpem=LimitingCurrentDensityMethod.Empirical,
        )

        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "H_+"].fix(7.38e-4)
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "OH_-"].fix(7.38e-4)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "H_+"].fix(7.38e-4)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "OH_-"].fix(7.38e-4)

        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)

        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Na_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Cl_-")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "H_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "OH_-")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "H2O")
        )

        m.fs.unit.water_trans_number_membrane["bpem"].fix((5.8 + 4.3) / 2)
        m.fs.unit.water_permeability_membrane["bpem"].fix((2.16e-14 + 1.75e-14) / 2)
        m.fs.unit.electrodes_resistance.fix(0)
        m.fs.unit.cell_pair_num.fix(5)
        m.fs.unit.current_utilization.fix(1)
        m.fs.unit.channel_height.fix(2.7e-4)
        m.fs.unit.membrane_areal_resistance["bpem"].fix((1.89e-4 + 1.77e-4) / 2)
        m.fs.unit.cell_width.fix(0.1)
        m.fs.unit.cell_length.fix(0.79)
        m.fs.unit.membrane_thickness["bpem"].fix(8e-4)
        m.fs.unit.ion_trans_number_membrane["bpem", "Na_+"].fix(0.5)
        m.fs.unit.ion_trans_number_membrane["bpem", "Cl_-"].fix(0.5)
        m.fs.unit.ion_trans_number_membrane["bpem", "H_+"].fix(0.1)
        m.fs.unit.ion_trans_number_membrane["bpem", "OH_-"].fix(0.1)

        m.fs.unit.diffus_mass["bpem"].fix((2.03 + 1.96) * 10**-9 / 2)
        m.fs.unit.membrane_fixed_charge["bpem"].fix(5e3)
        m.fs.unit.salt_conc_aem["bpem"].fix(2000)
        m.fs.unit.salt_conc_cem["bpem"].fix(2000)
        m.fs.unit.conc_water["bpem"].fix(50 * 1e3)
        m.fs.unit.kr["bpem"].fix(1.3 * 10**10)
        m.fs.unit.k2_zero["bpem"].fix(2 * 10**-6)
        m.fs.unit.relative_permittivity["bpem"].fix(30)
        m.fs.unit.membrane_fixed_catalyst_cem["bpem"].fix(5e3)
        m.fs.unit.membrane_fixed_catalyst_aem["bpem"].fix(5e3)
        m.fs.unit.k_a["bpem"].fix(3.5e2)
        m.fs.unit.k_b["bpem"].fix(5e4)

        # Set inlet stream. These have been scaled up by a factor of 1e3 from the ED process
        m.fs.unit.inlet_aem_side.pressure.fix(101325)
        m.fs.unit.inlet_aem_side.temperature.fix(298.15)
        m.fs.unit.inlet_cem_side.pressure.fix(101325)
        m.fs.unit.inlet_cem_side.temperature.fix(298.15)
        m.fs.unit.spacer_porosity.fix(1)

        # Set scaling for some critical inputs
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "H2O")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Na_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Cl_-")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "H_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "OH_-")
        )
        iscale.set_scaling_factor(m.fs.unit.k_a, 1e-2)
        iscale.set_scaling_factor(m.fs.unit.k_b, 1e-4)
        iscale.set_scaling_factor(m.fs.unit.flux_splitting, 1e3)

        for i in range(len(membrane_voltage)):
            m.fs.unit.potential_membrane_bpem[0].fix(membrane_voltage[i])

            # Experimental data corresponds to the MB-3 bipolar membrane from Wilhelm et al. (2002)
            # with additional modelling parameters obtaiend from simulation inputs from Maareev et al. (202)

            #  Negligible water splitting & limiting current
            if i == 0:
                iscale.calculate_scaling_factors(m.fs)
                assert degrees_of_freedom(m) == 0
                initialization_tester(m, outlvl=idaeslog.DEBUG)
                badly_scaled_var_values = {
                    var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
                }
                assert not badly_scaled_var_values
                results = solver.solve(m)
                assert_optimal_termination(results)
                current_density = m.fs.unit.current[0] / (
                    m.fs.unit.cell_width * m.fs.unit.cell_length
                )
                assert value(current_density) == pytest.approx(220, rel=2e-1)
                assert value(m.fs.unit.current_dens_lim_bpem[0]) == pytest.approx(
                    220, rel=2e-1
                )

            #  Start of water splitting
            if i == 1:
                iscale.calculate_scaling_factors(m.fs)
                assert degrees_of_freedom(m) == 0
                initialization_tester(m, outlvl=idaeslog.DEBUG)
                badly_scaled_var_values = {
                    var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
                }
                assert not badly_scaled_var_values
                results = solver.solve(m)
                assert_optimal_termination(results)
                current_density = m.fs.unit.current[0] / (
                    m.fs.unit.cell_width * m.fs.unit.cell_length
                )
                assert value(current_density) == pytest.approx(250, rel=2e-1)

            #  Water splitting regime
            if i == 2:
                iscale.calculate_scaling_factors(m.fs)
                assert degrees_of_freedom(m) == 0
                initialization_tester(m, outlvl=idaeslog.DEBUG)
                badly_scaled_var_values = {
                    var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
                }
                assert not badly_scaled_var_values
                results = solver.solve(m)
                assert_optimal_termination(results)
                current_density = m.fs.unit.current[0] / (
                    m.fs.unit.cell_width * m.fs.unit.cell_length
                )
                assert value(current_density) == pytest.approx(500, rel=2e-1)

            #  Significant water splitting
            if i == 3:
                iscale.calculate_scaling_factors(m.fs)
                assert degrees_of_freedom(m) == 0
                initialization_tester(m, outlvl=idaeslog.DEBUG)
                badly_scaled_var_values = {
                    var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
                }
                assert not badly_scaled_var_values
                results = solver.solve(m)
                assert_optimal_termination(results)
                current_density = m.fs.unit.current[0] / (
                    m.fs.unit.cell_width * m.fs.unit.cell_length
                )
                assert value(current_density) == pytest.approx(825, rel=2e-1)
