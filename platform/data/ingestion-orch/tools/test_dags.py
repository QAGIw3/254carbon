"""
Lightweight DAG validator without real Airflow.

Stubs minimal Airflow classes to import DAG modules and extract:
- dag_id, schedule_interval
- task ids
- simple dependencies via >> operator
"""
import importlib.util
import os
import sys
from types import ModuleType
from datetime import datetime, timedelta


# --- Minimal Airflow stubs ---
_ACTIVE_DAG = None


class _StubTask:
    def __init__(self, dag, task_id):
        self.task_id = task_id
        self.downstream = []
        self.upstream = []
        if dag is not None:
            dag._register_task(self)

    def __rshift__(self, other):
        # self >> other
        if isinstance(other, list):
            for t in other:
                self.__rshift__(t)
            return other
        self.downstream.append(other.task_id)
        other.upstream.append(self.task_id)
        return other


class DAG:
    def __init__(self, dag_id, default_args=None, description=None, schedule_interval=None, start_date=None, catchup=False):
        self.dag_id = dag_id
        self.default_args = default_args or {}
        self.description = description
        self.schedule_interval = schedule_interval
        self.start_date = start_date
        self.catchup = catchup
        self.tasks = {}

    def __enter__(self):
        global _ACTIVE_DAG
        _ACTIVE_DAG = self
        return self

    def __exit__(self, exc_type, exc, tb):
        global _ACTIVE_DAG
        _ACTIVE_DAG = None

    def _register_task(self, task: _StubTask):
        self.tasks[task.task_id] = task


class _PythonOperator(_StubTask):
    def __init__(self, task_id, python_callable=None, provide_context=False, **kwargs):
        super().__init__(_ACTIVE_DAG, task_id)
        self.python_callable = python_callable


class _Variable:
    @staticmethod
    def get(key, default_var=None):
        return os.getenv(key, default_var)


def days_ago(n):
    return datetime.utcnow() - timedelta(days=int(n))


def _install_airflow_stubs():
    airflow = ModuleType('airflow')
    airflow.DAG = DAG

    operators = ModuleType('airflow.operators')
    python = ModuleType('airflow.operators.python')
    python.PythonOperator = _PythonOperator
    operators.python = python

    utils = ModuleType('airflow.utils')
    dates = ModuleType('airflow.utils.dates')
    dates.days_ago = days_ago
    utils.dates = dates

    models = ModuleType('airflow.models')
    models.Variable = _Variable

    airflow.operators = operators
    airflow.utils = utils
    airflow.models = models

    sys.modules['airflow'] = airflow
    sys.modules['airflow.operators'] = operators
    sys.modules['airflow.operators.python'] = python
    sys.modules['airflow.utils'] = utils
    sys.modules['airflow.utils.dates'] = dates
    sys.modules['airflow.models'] = models

    # Also stub kafka so connectors importing KafkaProducer don't pull system kafka
    kafka = ModuleType('kafka')
    class KafkaProducer:  # noqa: N801
        def __init__(self, *a, **k): pass
        def send(self, *a, **k): pass
        def flush(self): pass
    kafka.KafkaProducer = KafkaProducer
    sys.modules['kafka'] = kafka

    # Stub connector modules referenced by these DAGs to avoid deep package imports
    ext_pkg = ModuleType('external')
    weather_pkg = ModuleType('external.weather')
    infra_pkg = ModuleType('external.infrastructure')
    econ_pkg = ModuleType('external.economics')

    class _StubConnector:
        def __init__(self, config):
            self.config = config
        def run(self):
            return 1

    noaa_mod = ModuleType('external.weather.noaa_cdo_connector')
    setattr(noaa_mod, 'NOAACDOConnector', _StubConnector)

    wb_energy_mod = ModuleType('external.infrastructure.world_bank_energy_connector')
    setattr(wb_energy_mod, 'WorldBankEnergyConnector', _StubConnector)

    wb_econ_mod = ModuleType('external.economics.world_bank_econ_connector')
    setattr(wb_econ_mod, 'WorldBankEconomicsConnector', _StubConnector)

    era5_mod = ModuleType('external.weather.era5_connector')
    setattr(era5_mod, 'ERA5Connector', _StubConnector)
    eia_mod = ModuleType('external.infrastructure.eia_connector')
    setattr(eia_mod, 'EIAOpenDataConnector', _StubConnector)
    entsoe_mod = ModuleType('external.infrastructure.entsoe_connector')
    setattr(entsoe_mod, 'ENTSOETransparencyConnector', _StubConnector)
    oim_mod = ModuleType('external.infrastructure.open_inframap_connector')
    setattr(oim_mod, 'OpenInfrastructureMapConnector', _StubConnector)
    oecd_mod = ModuleType('external.infrastructure.oecd_energy_connector')
    setattr(oecd_mod, 'OECDEnergyStatsConnector', _StubConnector)
    census_mod = ModuleType('external.demographics.us_census_connector')
    setattr(census_mod, 'USCensusConnector', _StubConnector)
    un_mod = ModuleType('external.demographics.un_data_connector')
    setattr(un_mod, 'UNDataConnector', _StubConnector)
    eurostat_mod = ModuleType('external.demographics.eurostat_connector')
    setattr(eurostat_mod, 'EurostatConnector', _StubConnector)
    cds_mod = ModuleType('external.weather.copernicus_cds_connector')
    setattr(cds_mod, 'CopernicusCDSConnector', _StubConnector)

    sys.modules['external'] = ext_pkg
    sys.modules['external.weather'] = weather_pkg
    sys.modules['external.infrastructure'] = infra_pkg
    sys.modules['external.economics'] = econ_pkg
    sys.modules['external.weather.noaa_cdo_connector'] = noaa_mod
    sys.modules['external.infrastructure.world_bank_energy_connector'] = wb_energy_mod
    sys.modules['external.economics.world_bank_econ_connector'] = wb_econ_mod
    sys.modules['external.weather.era5_connector'] = era5_mod
    sys.modules['external.infrastructure.eia_connector'] = eia_mod
    sys.modules['external.infrastructure.entsoe_connector'] = entsoe_mod
    sys.modules['external.infrastructure.open_inframap_connector'] = oim_mod
    sys.modules['external.infrastructure.oecd_energy_connector'] = oecd_mod
    sys.modules['external.demographics.us_census_connector'] = census_mod
    sys.modules['external.demographics.un_data_connector'] = un_mod
    sys.modules['external.demographics.eurostat_connector'] = eurostat_mod
    sys.modules['external.weather.copernicus_cds_connector'] = cds_mod


def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def summarize_dag(dag_module):
    dag = getattr(dag_module, 'dag', None)
    if dag is None:
        # Try to find a DAG instance attribute
        dag = next((v for v in dag_module.__dict__.values() if isinstance(v, DAG)), None)
    if dag is None:
        return {"error": "No DAG instance found"}
    tasks = list(dag.tasks.keys())
    edges = []
    for t in dag.tasks.values():
        for d in t.downstream:
            edges.append((t.task_id, d))
    return {
        "dag_id": dag.dag_id,
        "schedule_interval": dag.schedule_interval,
        "task_count": len(tasks),
        "tasks": tasks,
        "edges": edges,
    }


def main():
    _install_airflow_stubs()
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dags = [
        os.path.join(base, 'dags', 'noaa_cdo_ingestion_dag.py'),
        os.path.join(base, 'dags', 'world_bank_ingestion_dag.py'),
        os.path.join(base, 'dags', 'era5_ingestion_dag.py'),
        os.path.join(base, 'dags', 'copernicus_cds_ingestion_dag.py'),
        os.path.join(base, 'dags', 'eia_ingestion_dag.py'),
        os.path.join(base, 'dags', 'entsoe_ingestion_dag.py'),
        os.path.join(base, 'dags', 'open_inframap_ingestion_dag.py'),
        os.path.join(base, 'dags', 'oecd_energy_ingestion_dag.py'),
        os.path.join(base, 'dags', 'us_census_ingestion_dag.py'),
        os.path.join(base, 'dags', 'un_data_ingestion_dag.py'),
        os.path.join(base, 'dags', 'eurostat_ingestion_dag.py'),
    ]
    for p in dags:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            mod = load_module_from_path(name, p)
            summary = summarize_dag(mod)
            print(f"DAG {name}: {summary}")
        except Exception as e:
            print(f"DAG {name} failed to load: {e}")


if __name__ == '__main__':
    main()
