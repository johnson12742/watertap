# from pyomo.environ import ConcreteModel, Solverself.factory, TerminationCondition
import pytest
from idaes.core import FlowsheetBlock
from watertap.property_models.multicomp_aq_sol_prop_pack import MCASParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import DiagnosticsToolbox
from idaes.core.util.testing import initialization_tester
from pyomo.util.check_units import assert_units_consistent
import idaes.core.util.scaling as iscale
from pyomo.environ import (
    ConcreteModel,
    assert_optimal_termination,
    value,
    Set,
    Param,
    Var,
    Constraint,
    units as pyunits,
)

# import simple_prop_pack as props
from electrodialysis_bipolar_expt import (
    BiPolarElectrodialysis0D,
    LimitingCurrentDensityMethod,
    LimitingpotentialMethod,
    SaltmembraneMethod,
)

import idaes.logger as idaeslog
from watertap.core.solvers import get_solver

solver = get_solver()


class Test_operation_Method:
    factor = 1e0

    @pytest.fixture(scope="class")
    def bped_model(self):

        # solver = get_solver()

        # create model, flowsheet
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        # attach property package

        # "solute_list": ["Na_+", "Cl_-", "H_+", "OH_-"],
        # "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3, "H_+": 23e-3, "OH_-": 35.5e-3},
        # "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8, ("Liq", "H_+"): 5.19e-8,
        #                        ("Liq", "OH_-"): 7.92e-8},
        # "charge": {"Na_+": 1, "Cl_-": -1, "H_+": 1, "OH_-": -1},

        # "solute_list": ["Na_+", "Cl_-", "H_+", "OH_-"],
        # "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3, "H_+": 1e-3, "OH_-": 17.0e-3},
        # "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8, ("Liq", "H_+"): 36.23e-8,
        #                        ("Liq", "OH_-"): 20.64e-8},
        # "charge": {"Na_+": 1, "Cl_-": -1, "H_+": 1, "OH_-": -1},

        # "solute_list": ["Na_+", "Cl_-", "H_+", "OH_-"],
        # "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3, "H_+": 23e-3, "OH_-": 35.5e-3},
        # "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8, ("Liq", "H_+"): 5.19e-8,
        #                        ("Liq", "OH_-"): 7.92e-8},
        # "charge": {"Na_+": 1, "Cl_-": -1, "H_+": 1, "OH_-": -1},

        # "solute_list": ["Na_+", "Cl_-", "H_+", "OH_-"],
        # "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3, "H_+": 23e-3 / 1, "OH_-": 35.5e-3 / 1},
        # "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8,
        #                        ("Liq", "H_+"): 5.19e-8 * 1, ("Liq", "OH_-"): 7.92e-8 * 1},
        # "charge": {"Na_+": 1, "Cl_-": -1, "H_+": 1, "OH_-": -1},
        # "diffusivity_data": {("Liq", "Na_+"): 1.33e-9, ("Liq", "Cl_-"): 2.03e-9,
        #                      ("Liq", "H_+"): 1.33e-9, ("Liq", "OH_-"): 2.03e-9}

        # "solute_list": ["Na_+", "Cl_-", "H_+", "OH_-"],
        # "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3, "H_+": 1e-3, "OH_-": 17.0e-3},
        # "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8,
        #                        ("Liq", "H_+"): 36.23e-8, ("Liq", "OH_-"): 20.64e-8},
        # "charge": {"Na_+": 1, "Cl_-": -1, "H_+": 1, "OH_-": -1},
        # "diffusivity_data": {("Liq", "Na_+"): 1.33e-9, ("Liq", "Cl_-"): 2.03e-9,
        #                      ("Liq", "H_+"): 9.31e-9, ("Liq", "OH_-"): 5.27e-9}

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
            operation_mode="Constant_Current",
            Operation_method_bpem="Over_limiting",
            pressure_drop_method="none",
            has_catalyst=False,
            salt_membrane_method="input",
            salt_input_cem=500,
            salt_input_aem=500,
            limiting_current_density_method_bpem=LimitingCurrentDensityMethod.Empirical,
            limiting_current_density_data=112,
            limiting_potential_method_bpem=LimitingpotentialMethod.Empirical,
            limiting_potential_data=0.5,
        )
        # build a state block, must specify a time which by convention for steady state models is just 0
        # m.fs.stream = m.fs.properties.build_state_block([0], default={})

        # display the state block, it only has the state variables and they are all unfixed
        # print('\n---first display---')
        # m.fs.unit.display()
        # attempt to access properties so that they are built
        # m.fs.stream[0].mass_frac_phase_comp
        # after touching the property, the state block automatically builds it,
        # note the mass_frac_phase_comp variable and the constraint to calculate it
        print("\n---second display---")
        print("Degrees of freedom = ", degrees_of_freedom(m))
        print("config length = ", len(m.fs.unit.config))

        m.fs.unit.water_trans_number_membrane["bpem"].fix((5.8 + 4.3) / 2)
        # m.fs.unit.water_trans_number_membrane["aem"].fix(4.3)
        m.fs.unit.water_permeability_membrane["bpem"].fix((2.16e-14 + 1.75e-14) / 2)
        m.fs.unit.current.fix(10)
        m.fs.unit.electrodes_resistance.fix(0)
        m.fs.unit.cell_pair_num.fix(1)
        m.fs.unit.current_utilization.fix(1)
        m.fs.unit.channel_height.fix(2.7e-4)
        m.fs.unit.membrane_areal_resistance["bpem"].fix((1.89e-4 + 1.77e-4) / 2)
        m.fs.unit.cell_width.fix(0.1)
        m.fs.unit.cell_length.fix(0.79)
        m.fs.unit.membrane_thickness["bpem"].fix(1.3e-4)
        m.fs.unit.membrane_fixed_charge["bpem"].fix(1.5e3)
        m.fs.unit.diffus_mass["bpem"].fix((2.03 + 1.96) * 10**-9 / 2)
        m.fs.unit.solute_diffusivity_membrane["bpem", "Na_+"].fix(
            (1.8e-10 + 1.25e-10) / 2
        )
        m.fs.unit.solute_diffusivity_membrane["bpem", "Cl_-"].fix(
            (1.8e-10 + 1.25e-10) / 2
        )
        m.fs.unit.solute_diffusivity_membrane["bpem", "H_+"].fix(
            (1.8e-10 + 1.25e-10) / 2
        )
        m.fs.unit.solute_diffusivity_membrane["bpem", "OH_-"].fix(
            (1.8e-10 + 1.25e-10) / 2
        )
        m.fs.unit.ion_trans_number_membrane["bpem", "Na_+"].fix(0.5)
        m.fs.unit.ion_trans_number_membrane["bpem", "Cl_-"].fix(0.5)
        m.fs.unit.ion_trans_number_membrane["bpem", "H_+"].fix(0.1)
        m.fs.unit.ion_trans_number_membrane["bpem", "OH_-"].fix(0.1)
        m.fs.unit.kr["bpem"].fix(1.3 * 10**11)
        m.fs.unit.kd_zero["bpem"].fix(2 * 10**-5)
        m.fs.unit.relative_permittivity["bpem"].fix(20)

        # assert (
        #         sum(
        #             value(m.fs.unit.ion_trans_number_membrane["cem", j])
        #             for j in m.fs.properties.ion_set
        #         )
        #         == 1
        # )
        # assert (
        #         sum(
        #             value(m.fs.unit.ion_trans_number_membrane["aem", j])
        #             for j in m.fs.properties.ion_set
        #         )
        #         == 1
        # )
        # assert sum(
        #     value(m.fs.unit.ion_trans_number_membrane["cem", j])
        #     for j in m.fs.properties.cation_set
        # ) == sum(
        #     value(m.fs.unit.ion_trans_number_membrane["aem", j])
        #     for j in m.fs.properties.anion_set
        # )

        m.fs.unit.inlet_aem_side.pressure.fix(101325)
        m.fs.unit.inlet_aem_side.temperature.fix(298.15)
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "H2O"].fix(
            2.40e-1 * self.factor
        )
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(
            7.38e-4 * self.factor
        )
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(
            7.38e-4 * self.factor
        )
        # m.fs.unit.inlet_aem_side.conc_mol_phase_comp[0, "Liq", "Na_+"].fix(500)
        # m.fs.unit.inlet_aem_side.conc_mol_phase_comp[0, "Liq", "Cl_-"].fix(500)
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "H_+"].fix(
            7.38e-4 * self.factor
        )
        m.fs.unit.inlet_aem_side.flow_mol_phase_comp[0, "Liq", "OH_-"].fix(
            7.38e-4 * self.factor
        )
        m.fs.unit.inlet_cem_side.pressure.fix(101325)
        m.fs.unit.inlet_cem_side.temperature.fix(298.15)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "H2O"].fix(
            2.40e-1 * self.factor
        )
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(
            7.38e-4 * self.factor
        )
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(
            7.38e-4 * self.factor
        )
        # m.fs.unit.inlet_cem_side.conc_mol_phase_comp[0, "Liq", "Na_+"].fix(500)
        # m.fs.unit.inlet_cem_side.conc_mol_phase_comp[0, "Liq", "Cl_-"].fix(500)
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "H_+"].fix(
            7.38e-4 * self.factor
        )
        m.fs.unit.inlet_cem_side.flow_mol_phase_comp[0, "Liq", "OH_-"].fix(
            7.38e-4 * self.factor
        )
        m.fs.unit.spacer_porosity.fix(1)
        return m

    @pytest.mark.unit
    def test_assign(self, bped_model):
        m = bped_model
        print("\n---third display---")
        # m.fs.unit.display()
        print("Degrees of freedom = ", degrees_of_freedom(m))
        assert degrees_of_freedom(m) == 0

        print("config length = ", len(m.fs.unit.config))

        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3 * self.factor**-1, index=("Liq", "H2O")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3 * 1**-1, index=("Liq", "Na_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3 * 1**-1, index=("Liq", "Cl_-")
        )
        # m.fs.properties.set_default_scaling("conc_mol_phase_comp", 2e-3, index=("Liq", "Na_+"))
        # m.fs.properties.set_default_scaling("conc_mol_phase_comp", 2e-3, index=("Liq", "Cl_-"))
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3 * 1**-1, index=("Liq", "H_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3 * 1**-1, index=("Liq", "OH_-")
        )
        # print("420")
        # @pytest.mark.unit
        #     def test_assign(self):
        # iscale.calculate_scaling_self.factors(m.fs)
        iscale.calculate_scaling_factors(m.fs)
        # print("69")

        dt = DiagnosticsToolbox(m)
        dt.report_structural_issues()
        dt.display_potential_evaluation_errors()

        initialization_tester(m, outlvl=idaeslog.DEBUG)
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
        }
        print("badly_scaled_var_values =", badly_scaled_var_values)
        # assert not badly_scaled_var_values
        # check to make sure DOF does not change
        assert degrees_of_freedom(m) == 0
        # m.fs.unit.spacer_porosity.unfix
        results = solver.solve(m)
        assert_optimal_termination(results)

        print("\n---fourth display---")
        m.fs.unit.display()
        # assert results.solver.termination_condition == TerminationCondition.optimal
        print(m.fs.unit.nonelec_flux_in.display())
        print(m.fs.unit.elec_migration_flux_out.display())
        # print(m.fs.unit.aem_side.mass_transfer_term.display())
        print(m.fs.unit.current_dens_lim_bpem.display())
        print(m.fs.unit.potential_lim_bpem.display())
        print(m.fs.unit.aem_side.mass_transfer_term.display())
        print(m.fs.unit.cem_side.mass_transfer_term.display())
        print(m.fs.unit.salt_conc_aem.display())
        print(m.fs.unit.salt_conc_cem.display())
        # print(m.fs.unit.cem_side.properties_in[0].conc_mol_phase_comp.display())
        # print(m.fs.unit.cem_side.properties_out[0].conc_mol_phase_comp.display())
        # print(m.fs.unit.aem_side.properties_in[0].conc_mol_phase_comp.display())
        # print(m.fs.unit.aem_side.properties_out[0].conc_mol_phase_comp.display())


# limiting_current_density_method=LimitingCurrentDensityMethod.Theoretical


# m.fs.stream[0].display()
#
# # touch another property
# m.fs.stream[0].conc_mass_phase_comp
# # after touching this property, the state block automatically builds it AND any other properties that are necessary,
# # note that now there is the conc_mass_phase_comp and dens_mass_phase variable and associated constraints
# print('\n---third display---')
# m.fs.stream[0].display()
#
# # touch last property
# m.fs.stream[0].flow_vol_phase
#
# # now that we have a state block, we can fix the state variables and solve for the properties
# m.fs.stream[0].temperature.fix(273.15 + 25)
# m.fs.stream[0].pressure.fix(101325)
# m.fs.stream[0].flow_mass_phase_comp['Liq', 'H2O'].fix(1)
# m.fs.stream[0].flow_mass_phase_comp['Liq', 'NaCl'].fix(0.035)
# m.fs.stream[0].flow_mass_phase_comp['Liq', 'TSS'].fix(120e-6)
#
# # the user should provide the scale for the flow rate, so that our tools can ensure the model is well scaled
# # generally scaling self.factors should be such that if it is multiplied by the variable it will range between 0.01 and 100
# m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1, index=('Liq', 'H2O'))
# m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1e2, index=('Liq', 'NaCl'))
# m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1e4, index=('Liq', 'TSS'))
# iscale.calculate_scaling_self.factors(m.fs)  # this utility scales the model
#
# # solving
# assert_units_consistent(m)  # check that units are consistent
# assert(degrees_of_freedom(m) == 0)  # check that the degrees of freedom are what we expect
#
# solver = Solverself.factory('ipopt')
# solver.options = {'tol': 1e-8, 'nlp_scaling_method': 'user-scaling'}
#
# results = solver.solve(m, tee=False)
# assert results.solver.termination_condition == TerminationCondition.optimal
#
# # display results
# print('\n---fourth display---')
# m.fs.stream[0].display()
# # note that the properties are solved, and the body of the constraints are small (residual)
#
# # equation oriented modeling has several advantages, one of them is that we can unfix variables and fix others
# # instead of setting the mass flow rates, we can set the volumetric flow rate and mass fractions
# m.fs.stream[0].flow_mass_phase_comp['Liq', 'H2O'].unfix()
# m.fs.stream[0].flow_mass_phase_comp['Liq', 'NaCl'].unfix()
# m.fs.stream[0].flow_mass_phase_comp['Liq', 'TSS'].unfix()
#
# m.fs.stream[0].flow_vol_phase['Liq'].fix(1.5e-3)
# m.fs.stream[0].mass_frac_phase_comp['Liq', 'NaCl'].fix(0.05)
# m.fs.stream[0].mass_frac_phase_comp['Liq', 'TSS'].fix(80e-6)
#
# # resolve
# results = solver.solve(m, tee=False)
# assert results.solver.termination_condition == TerminationCondition.optimal
#
# print('\n---fifth display---')
# m.fs.stream[0].display()
