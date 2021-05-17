from dku_plugin_test_utils import dss_scenario

project_key = "Plugin_Test_TsPrep"


def test_resampling(user_dss_clients):
    dss_scenario.run(client=user_dss_clients, project_key=project_key, scenario_id="Run_resampling")


def test_windowing(user_dss_clients):
    dss_scenario.run(client=user_dss_clients, project_key=project_key, scenario_id="Windowing")


def test_extrema_extraction(user_dss_clients):
    dss_scenario.run(client=user_dss_clients, project_key=project_key, scenario_id="Extrema")


def test_interval_extraction(user_dss_clients):
    dss_scenario.run(client=user_dss_clients, project_key=project_key, scenario_id="Intervals")


def test_decomposition(user_dss_clients):
    dss_scenario.run(client=user_dss_clients, project_key=project_key, scenario_id="Decomposition")