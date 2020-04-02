import nifty
import nifty.graph.opt.lifted_multicut as nlmc
import nifty.graph.opt.multicut as nmc
import nifty.graph.rag as nrag
import numpy as np


# from ...features import probs_to_costs, get_z_edges


# TODO time-limit
def lifted_multicut(n_nodes, local_uvs, local_costs, lifted_uvs, lifted_costs, time_limit=None):
    graph = nifty.graph.UndirectedGraph(n_nodes)
    graph.insertEdges(local_uvs)


    lifted_obj = nlmc.liftedMulticutObjective(graph)
    lifted_obj.setCosts(local_uvs, local_costs)
    lifted_obj.setCosts(lifted_uvs, lifted_costs)
    # visitor = lifted_obj.verboseVisitor(100)
    solver_ehc = lifted_obj.liftedMulticutGreedyAdditiveFactory().create(lifted_obj)
    node_labels = solver_ehc.optimize()
    solver_kl = lifted_obj.liftedMulticutKernighanLinFactory().create(lifted_obj)
    node_labels = solver_kl.optimize(node_labels)
    return node_labels


def multicut(rag, n_nodes, uvs, costs, time_limit=None,
             solver_type='fusionMoves',
             verbose_visitNth=100000000):
    MulticutObjective = rag.MulticutObjective

    # graph = nifty.graph.UndirectedGraph(n_nodes)
    # graph.insertEdges(uvs)
    obj_mc = MulticutObjective(rag, costs)


    if solver_type == 'multicutIlpCplex':
        solver = MulticutObjective.multicutIlpCplexFactory().create(obj_mc)

        if time_limit is not None:
            visitor = obj_mc.verboseVisitor(timeLimitSolver=time_limit)
            node_labels = solver.optimize(visitor=visitor)
        else:
            visitor = obj_mc.verboseVisitor()
            node_labels = solver.optimize(visitor=visitor)

    elif solver_type == 'kernighanLin':
        solverFactory = obj_mc.kernighanLinFactory()
        solver = solverFactory.create(obj_mc)
        visitor = obj_mc.verboseVisitor(1)
        node_labels = solver.optimize(visitor)
    elif solver_type == 'GAEC+kernighanLin':
        solverFactory = obj_mc.greedyAdditiveFactory()
        solver = solverFactory.create(obj_mc)
        visitor = obj_mc.verboseVisitor(1)
        arg = solver.optimize()

        solverFactory = obj_mc.kernighanLinFactory()
        solver = solverFactory.create(obj_mc)
        visitor = obj_mc.verboseVisitor(1)
        node_labels = solver.optimize(visitor, arg)

    elif solver_type == 'fusionMoves':
        """
        .def_readwrite("nodeNumStopCond", &SettingsType::nodeNumStopCond)
        .def_readwrite("weightStopCond", &SettingsType::weightStopCond)
        .def_readwrite("visitNth", &SettingsType::visitNth)
        """
        solverFactory = obj_mc.greedyAdditiveFactory()
        solver = solverFactory.create(obj_mc)

        # TODO: visitNth???
        """
        py::arg("visitNth")=1, 
        py::arg("timeLimitSolver") = std::numeric_limits<double>::infinity(),
        py::arg("timeLimitTotal") = std::numeric_limits<double>::infinity(),
        py::arg("logLevel") = nifty::logging::LogLevel::WARN
        nifty.LogLevel
            .value("NONE", nifty::logging::LogLevel::NONE)
            .value("FATAL", nifty::logging::LogLevel::FATAL)
            .value("ERROR", nifty::logging::LogLevel::ERROR)
            .value("WARN", nifty::logging::LogLevel::WARN)
            .value("INFO", nifty::logging::LogLevel::INFO)
            .value("DEBUG", nifty::logging::LogLevel::DEBUG)
            .value("TRACE", nifty::logging::LogLevel::TRACE)
        ----
        .def("stopOptimize",&VisitorType::stopOptimize)
        .def_property_readonly("timeLimitSolver", &VisitorType::timeLimitSolver)
        .def_property_readonly("timeLimitTotal", &VisitorType::timeLimitTotal)
        .def_property_readonly("runtimeSolver", &VisitorType::runtimeSolver)
        .def_property_readonly("runtimeTotal", &VisitorType::runtimeTotal)
        """
        visitor = obj_mc.verboseVisitor(1)
        arg = solver.optimize()

        """
        .def_readwrite("numberOfInnerIterations", &SettingsType::numberOfInnerIterations)
        .def_readwrite("numberOfOuterIterations", &SettingsType::numberOfOuterIterations)
        .def_readwrite("epsilon", &SettingsType::epsilon)
        """
        solverFactory = obj_mc.kernighanLinFactory()
        solver = solverFactory.create(obj_mc)
        visitor = obj_mc.verboseVisitor(1)
        arg2 = solver.optimize(visitor, arg)

        """
        .def_readwrite("sigma", &PGenSettigns::sigma)
        .def_readwrite("numberOfSeeds", &PGenSettigns::numberOfSeeds)
        .def_readwrite("seedingStrategie", &PGenSettigns::seedingStrategie)
            .value("SEED_FROM_NEGATIVE", SeedingStrategie::SEED_FROM_NEGATIVE)
            .value("SEED_FROM_ALL", SeedingStrategie::SEED_FROM_ALL)
        """
        pgen = obj_mc.watershedCcProposals(sigma=1.0, numberOfSeeds=0.1)

        """
        .def_readwrite("proposalGenerator", &SettingsType::proposalGeneratorFactory)
        .def_readwrite("numberOfThreads", &SettingsType::numberOfThreads)
        .def_readwrite("numberOfIterations",&SettingsType::numberOfIterations)
        .def_readwrite("stopIfNoImprovement",&SettingsType::stopIfNoImprovement)
        .def_readwrite("fusionMoveSettings",&SettingsType::fusionMoveSettings)
            // the fusion mover parameter itself
            .def_readwrite("mcFactory",&FusionMoveSettings::mcFactory)
        """
        solverFactory = obj_mc.ccFusionMoveBasedFactory(proposalGenerator=pgen, numberOfIterations=100,
                                                        stopIfNoImprovement=10)

        solver = solverFactory.create(obj_mc)
        visitor = obj_mc.verboseVisitor(1)
        node_labels = solver.optimize(visitor, arg2)

    else:
        raise NotImplementedError()

    return node_labels
