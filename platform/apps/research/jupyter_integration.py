"""
Jupyter Notebook Integration for Research Platform

Integration system for Jupyter notebooks in the research environment:
- Notebook execution and scheduling
- Data access APIs for notebooks
- Model experimentation tracking
- Results visualization and sharing
- Collaborative research features
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import uuid
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class JupyterIntegration:
    """
    Jupyter notebook integration for research workflows.

    Features:
    - Notebook execution management
    - Data API integration
    - Experiment tracking
    - Results sharing
    """

    def __init__(self, notebooks_dir: str = "/research/notebooks"):
        self.notebooks_dir = Path(notebooks_dir)
        self.notebooks_dir.mkdir(parents=True, exist_ok=True)

        self.execution_history = {}
        self.experiment_registry = ExperimentRegistry()

    def create_research_notebook(
        self,
        title: str,
        description: str,
        template: str = "research_template",
        tags: List[str] = None,
        author: str = "researcher"
    ) -> str:
        """
        Create a new research notebook from template.

        Args:
            title: Notebook title
            description: Notebook description
            template: Template to use
            tags: Research tags
            author: Author name

        Returns:
            Notebook ID
        """
        notebook_id = f"NB_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create notebook structure
        notebook_data = {
            'id': notebook_id,
            'title': title,
            'description': description,
            'template': template,
            'tags': tags or [],
            'author': author,
            'created_at': datetime.now(),
            'status': 'created',
            'cells': self._create_notebook_cells(template),
            'metadata': {
                'kernelspec': {
                    'display_name': 'Python 3 (254Carbon)',
                    'language': 'python',
                    'name': 'python3'
                },
                'language_info': {
                    'name': 'python',
                    'version': '3.11.0'
                }
            }
        }

        # Save notebook file
        notebook_path = self.notebooks_dir / f"{notebook_id}.ipynb"
        with open(notebook_path, 'w') as f:
            json.dump(self._format_notebook(notebook_data), f, indent=2)

        logger.info(f"Created research notebook: {notebook_id}")
        return notebook_id

    def _create_notebook_cells(self, template: str) -> List[Dict[str, Any]]:
        """Create notebook cells based on template."""
        if template == "research_template":
            return [
                {
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': [
                        '# Research Notebook Template\n',
                        '\n',
                        'This notebook provides a template for energy market research using the 254Carbon platform.\n',
                        '\n',
                        '## Setup and Data Access\n',
                        'Import required libraries and connect to 254Carbon APIs.'
                    ]
                },
                {
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': [
                        'import pandas as pd\n',
                        'import numpy as np\n',
                        'import matplotlib.pyplot as plt\n',
                        'from datetime import datetime, timedelta\n',
                        '\n',
                        '# 254Carbon API imports\n',
                        'from carbon254 import APIClient\n',
                        'from carbon254.research import ResearchTools\n',
                        '\n',
                        '# Initialize API client\n',
                        'api = APIClient()\n',
                        'research = ResearchTools(api)\n',
                        '\n',
                        'print("254Carbon research environment initialized")'
                    ]
                },
                {
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': [
                        '## Data Retrieval\n',
                        'Access historical and real-time energy market data.'
                    ]
                },
                {
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': [
                        '# Example: Get oil price data\n',
                        'oil_prices = api.get_prices(\n',
                        '    commodity="WTI",\n',
                        '    start_date="2024-01-01",\n',
                        '    end_date="2024-12-31"\n',
                        ')\n',
                        '\n',
                        '# Example: Get carbon price data\n',
                        'carbon_prices = api.get_prices(\n',
                        '    commodity="EUA",\n',
                        '    start_date="2024-01-01",\n',
                        '    end_date="2024-12-31"\n',
                        ')\n',
                        '\n',
                        'print(f"Oil data shape: {oil_prices.shape}")\n',
                        'print(f"Carbon data shape: {carbon_prices.shape}")'
                    ]
                }
            ]

        return []  # Default empty cells

    def _format_notebook(self, notebook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format notebook data for Jupyter format."""
        return {
            'cells': notebook_data['cells'],
            'metadata': notebook_data['metadata'],
            'nbformat': 4,
            'nbformat_minor': 5
        }

    def execute_notebook(
        self,
        notebook_id: str,
        execution_timeout: int = 300,
        max_memory: str = "2GB"
    ) -> Dict[str, Any]:
        """
        Execute a research notebook.

        Args:
            notebook_id: Notebook ID to execute
            execution_timeout: Execution timeout in seconds
            max_memory: Maximum memory allocation

        Returns:
            Execution results
        """
        notebook_path = self.notebooks_dir / f"{notebook_id}.ipynb"

        if not notebook_path.exists():
            return {'error': f'Notebook {notebook_id} not found'}

        try:
            # Execute notebook using nbconvert or papermill
            execution_id = f"EXEC_{uuid.uuid4().hex[:8]}"

            # In production: Use papermill for parameterized execution
            result = subprocess.run([
                'python', '-m', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--ExecutePreprocessor.timeout=300',
                str(notebook_path)
            ], capture_output=True, text=True, timeout=execution_timeout)

            execution_result = {
                'execution_id': execution_id,
                'notebook_id': notebook_id,
                'status': 'completed' if result.returncode == 0 else 'failed',
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': datetime.now(),
                'timeout': execution_timeout
            }

            # Store execution history
            self.execution_history[execution_id] = execution_result

            return execution_result

        except subprocess.TimeoutExpired:
            return {
                'execution_id': execution_id,
                'notebook_id': notebook_id,
                'status': 'timeout',
                'error': f'Execution timed out after {execution_timeout} seconds'
            }
        except Exception as e:
            return {
                'execution_id': execution_id,
                'notebook_id': notebook_id,
                'status': 'error',
                'error': str(e)
            }

    def get_notebook_results(self, execution_id: str) -> Dict[str, Any]:
        """Get results from notebook execution."""
        return self.execution_history.get(execution_id, {'error': 'Execution not found'})

    def create_experiment_notebook(
        self,
        experiment_name: str,
        model_type: str,
        dataset: str,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Create notebook for model experimentation.

        Args:
            experiment_name: Name of the experiment
            model_type: Type of model being tested
            dataset: Dataset being used
            parameters: Model parameters

        Returns:
            Notebook ID
        """
        experiment_id = self.experiment_registry.register_experiment(
            name=experiment_name,
            model_type=model_type,
            dataset=dataset,
            parameters=parameters
        )

        # Create experiment notebook
        notebook_id = self.create_research_notebook(
            title=f"Experiment: {experiment_name}",
            description=f"Model experimentation for {model_type} on {dataset}",
            template="experiment_template",
            tags=["experiment", model_type, dataset],
            author="researcher"
        )

        # Add experiment metadata
        notebook_path = self.notebooks_dir / f"{notebook_id}.ipynb"
        with open(notebook_path, 'r') as f:
            notebook_data = json.load(f)

        # Add experiment cell
        experiment_cell = {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# Experiment Setup\n',
                f'experiment_id = "{experiment_id}"\n',
                f'model_type = "{model_type}"\n',
                f'dataset = "{dataset}"\n',
                'parameters = ' + str(parameters) + '\n',
                '\n',
                '# Log experiment start\n',
                'research.log_experiment_start(experiment_id, parameters)\n',
                '\n',
                'print(f"Starting experiment {experiment_id}")'
            ]
        }

        notebook_data['cells'].insert(1, experiment_cell)

        # Save updated notebook
        with open(notebook_path, 'w') as f:
            json.dump(notebook_data, f, indent=2)

        return notebook_id


class ExperimentRegistry:
    """Registry for tracking model experiments."""

    def __init__(self, registry_file: str = "/research/experiments/registry.json"):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

        self.experiments = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load experiment registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_registry(self) -> None:
        """Save experiment registry to file."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)

    def register_experiment(
        self,
        name: str,
        model_type: str,
        dataset: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Register a new experiment."""
        experiment_id = f"EXP_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        experiment = {
            'id': experiment_id,
            'name': name,
            'model_type': model_type,
            'dataset': dataset,
            'parameters': parameters,
            'created_at': datetime.now(),
            'status': 'registered',
            'results': {},
            'execution_time': None,
            'notebook_id': None
        }

        self.experiments[experiment_id] = experiment
        self._save_registry()

        logger.info(f"Registered experiment: {experiment_id}")
        return experiment_id

    def update_experiment_results(
        self,
        experiment_id: str,
        results: Dict[str, Any],
        execution_time: Optional[float] = None
    ) -> None:
        """Update experiment results."""
        if experiment_id not in self.experiments:
            logger.warning(f"Experiment {experiment_id} not found")
            return

        self.experiments[experiment_id]['results'] = results
        self.experiments[experiment_id]['status'] = 'completed'
        self.experiments[experiment_id]['execution_time'] = execution_time
        self.experiments[experiment_id]['completed_at'] = datetime.now()

        self._save_registry()
        logger.info(f"Updated results for experiment: {experiment_id}")

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        if not self.experiments:
            return {'total_experiments': 0, 'status_distribution': {}}

        status_counts = {}
        for exp in self.experiments.values():
            status = exp['status']
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            'total_experiments': len(self.experiments),
            'status_distribution': status_counts,
            'recent_experiments': list(self.experiments.keys())[-5:],  # Last 5 experiments
            'model_types': list(set(exp['model_type'] for exp in self.experiments.values())),
            'datasets': list(set(exp['dataset'] for exp in self.experiments.values()))
        }

    def get_experiment_details(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed information about an experiment."""
        return self.experiments.get(experiment_id, {'error': 'Experiment not found'})


class ResearchDataAPI:
    """API for accessing research data in notebooks."""

    def __init__(self, api_client):
        self.api = api_client

    def get_historical_prices(
        self,
        commodities: List[str],
        start_date: str,
        end_date: str,
        frequency: str = 'daily'
    ) -> pd.DataFrame:
        """Get historical price data for research."""
        # In production: Implement actual API calls
        return pd.DataFrame()  # Placeholder

    def get_fundamentals_data(
        self,
        fundamentals: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Get fundamental data for research."""
        # In production: Implement actual API calls
        return pd.DataFrame()  # Placeholder

    def get_weather_data(
        self,
        locations: List[str],
        start_date: str,
        end_date: str,
        variables: List[str] = None
    ) -> pd.DataFrame:
        """Get weather data for research."""
        # In production: Implement actual API calls
        return pd.DataFrame()  # Placeholder

    def get_carbon_data(
        self,
        markets: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Get carbon market data for research."""
        # In production: Implement actual API calls
        return pd.DataFrame()  # Placeholder


class ResearchVisualization:
    """Visualization tools for research notebooks."""

    def __init__(self):
        self.plot_themes = {
            'research': {
                'figure.figsize': (12, 8),
                'axes.grid': True,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10
            }
        }

    def create_price_chart(
        self,
        price_data: pd.DataFrame,
        title: str = "Price Chart",
        save_path: Optional[str] = None
    ) -> str:
        """Create price visualization chart."""
        # In production: Generate actual charts
        return f"Chart created: {title}"

    def create_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Correlation Heatmap",
        save_path: Optional[str] = None
    ) -> str:
        """Create correlation heatmap."""
        # In production: Generate actual heatmaps
        return f"Heatmap created: {title}"

    def create_performance_dashboard(
        self,
        portfolio_data: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """Create portfolio performance dashboard."""
        # In production: Generate actual dashboards
        return "Dashboard created"
