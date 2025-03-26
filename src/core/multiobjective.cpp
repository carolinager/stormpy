#include "multiobjective.h"

#include "storm/modelchecker/multiobjective/multiObjectiveModelChecking.h"

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/environment/Environment.h"
#include "storm/environment/modelchecker/MultiObjectiveModelCheckerEnvironment.h"
#include "storm/modelchecker/multiobjective/constraintbased/SparseCbAchievabilityQuery.h"
#include "storm/modelchecker/multiobjective/deterministicScheds/DeterministicSchedsAchievabilityChecker.h"
#include "storm/modelchecker/multiobjective/deterministicScheds/DeterministicSchedsParetoExplorer.h"
#include "storm/modelchecker/multiobjective/pcaa/SparsePcaaAchievabilityQuery.h"
#include "storm/modelchecker/multiobjective/pcaa/SparsePcaaParetoQuery.h"
#include "storm/modelchecker/multiobjective/pcaa/SparsePcaaQuantitativeQuery.h"
#include "storm/modelchecker/multiobjective/pcaa/StandardMdpPcaaWeightVectorChecker.h"
#include "storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h"
#include "storm/models/sparse/MarkovAutomaton.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/models/sparse/StandardRewardModel.h"
#include "storm/settings/SettingsManager.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/storage/BitVector.h"
#include "storm/settings/modules/CoreSettings.h"
#include "storm/utility/Stopwatch.h"
#include "storm/utility/macros.h"

#include "storm/exceptions/InvalidArgumentException.h"
#include "storm/exceptions/InvalidEnvironmentException.h"


template <typename ValueType>
std::tuple<ValueType, ValueType, storm::storage::Scheduler<ValueType>> _checkForRelReach(storm::Environment const& env, storm::models::sparse::Mdp<ValueType> & model, uint64_t state,
                                                                storm::logic::MultiObjectiveFormula const& formula, std::vector<ValueType> weightVector, bool computeScheduler) {
    // This internal check for RelReach requires some cleanup before it makes sense to merge this.
    //    STORM_LOG_ASSERT(model.getInitialStates().getNumberOfSetBits() == 1,
    //                     "Multi-objective Model checking on model with multiple initial states is not supported.");

    // Preprocess the model and make the passed state the only initial state
    storm::storage::BitVector oldInit = model.getInitialStates();
    storm::storage::BitVector newInit(model.getNumberOfStates());
    newInit.set(state, true);
    model.setInitialStates(newInit); // this changes the model! but we reset to the old initial states
    auto preprocessorResult = storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<storm::models::sparse::Mdp<ValueType>>::preprocess(env, model, formula);

    auto checker = storm::modelchecker::multiobjective::StandardMdpPcaaWeightVectorChecker(preprocessorResult);
    checker.check(env, weightVector);
    model.setInitialStates(oldInit);

    // currently, the following is not necessary in practice since over and underApprox are the same anyways, but in theory it would not be sound otherwise
    ValueType underApprox = 0;
    ValueType overApprox = 0;
    for (int i = 0; i < weightVector.size(); ++i) {
        if (weightVector[i] >=0) {
            underApprox += weightVector[i] * checker.getUnderApproximationOfInitialStateResults().at(i);
            overApprox += weightVector[i] * checker.getOverApproximationOfInitialStateResults().at(i);
        } else {
            underApprox += weightVector[i] * checker.getOverApproximationOfInitialStateResults().at(i);
            overApprox += weightVector[i] * checker.getUnderApproximationOfInitialStateResults().at(i);
        }
    }

    if (computeScheduler) {
        return {underApprox, overApprox, checker.computeScheduler()};
    } else {
        return {underApprox, overApprox, storm::storage::Scheduler<ValueType>(0)};
    }
}

// Define python bindings
void define_multiobjective(py::module& m) {
    m.def("compute_rel_reach_helper", &_checkForRelReach<double>, py::arg("env"), py::arg("model"), py::arg("state"), py::arg("formula"), py::arg("weightVector"), py::arg("compute_scheduler")=false);
    m.def("compute_rel_reach_helper_exact", &_checkForRelReach<storm::RationalNumber>, py::arg("env"), py::arg("model"), py::arg("state"), py::arg("formula"), py::arg("weightVector"), py::arg("compute_scheduler")=false);
}
