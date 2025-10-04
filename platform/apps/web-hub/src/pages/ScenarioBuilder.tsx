import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AlertCircle, Play, Trash2, Plus, Minus } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { apiClient } from '@/services/api';

interface ScenarioAssumption {
  id: string;
  name: string;
  type: 'percentage' | 'multiplier' | 'absolute' | 'distribution';
  value: number;
  distribution?: 'normal' | 'uniform' | 'beta';
  std_dev?: number;
  min?: number;
  max?: number;
  alpha?: number;
  beta?: number;
}

interface ScenarioSpec {
  name: string;
  description: string;
  category: 'demand' | 'supply' | 'policy' | 'economic' | 'weather';
  assumptions: ScenarioAssumption[];
  time_horizon: {
    start: string;
    end: string;
  };
  markets: string[];
  regions: string[];
  run_config: {
    parallel_runs: number;
    output_granularity: 'hourly' | 'daily' | 'monthly' | 'quarterly';
    include_confidence_intervals: boolean;
  };
}

interface ScenarioRun {
  run_id: string;
  scenario_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress: number;
  started_at: string;
  completed_at?: string;
  results?: any;
}

export default function ScenarioBuilder() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState('create');

  // Form state for scenario creation/editing
  const [scenarioForm, setScenarioForm] = useState<ScenarioSpec>({
    name: '',
    description: '',
    category: 'demand',
    assumptions: [],
    time_horizon: {
      start: new Date().toISOString().split('T')[0],
      end: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
    },
    markets: ['power'],
    regions: ['MISO'],
    run_config: {
      parallel_runs: 100,
      output_granularity: 'monthly',
      include_confidence_intervals: true
    }
  });

  // Query for existing scenarios
  const { data: scenarios, isLoading: scenariosLoading } = useQuery({
    queryKey: ['scenarios'],
    queryFn: () => apiClient.get('/scenarios').then(res => res.data.scenarios || [])
  });

  // Query for scenario runs
  const { data: runs, isLoading: runsLoading } = useQuery({
    queryKey: ['scenario-runs'],
    queryFn: () => apiClient.get('/scenarios/runs').then(res => res.data.runs || [])
  });

  // Mutations
  const createScenarioMutation = useMutation({
    mutationFn: (scenario: ScenarioSpec) => apiClient.post('/scenarios', scenario),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenarios'] });
      setActiveTab('manage');
    }
  });

  const executeScenarioMutation = useMutation({
    mutationFn: (scenarioId: string) =>
      apiClient.post(`/scenarios/${scenarioId}/execute`, {
        scenario_id: scenarioId,
        spec: scenarioForm,
        priority: 'normal',
        notify_on_completion: false
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenario-runs'] });
    }
  });

  const deleteScenarioMutation = useMutation({
    mutationFn: (scenarioId: string) => apiClient.delete(`/scenarios/${scenarioId}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['scenarios'] })
  });

  // Add new assumption
  const addAssumption = () => {
    const newAssumption: ScenarioAssumption = {
      id: `assumption-${Date.now()}`,
      name: '',
      type: 'percentage',
      value: 0
    };
    setScenarioForm(prev => ({
      ...prev,
      assumptions: [...prev.assumptions, newAssumption]
    }));
  };

  // Remove assumption
  const removeAssumption = (id: string) => {
    setScenarioForm(prev => ({
      ...prev,
      assumptions: prev.assumptions.filter(a => a.id !== id)
    }));
  };

  // Update assumption
  const updateAssumption = (id: string, updates: Partial<ScenarioAssumption>) => {
    setScenarioForm(prev => ({
      ...prev,
      assumptions: prev.assumptions.map(a =>
        a.id === id ? { ...a, ...updates } : a
      )
    }));
  };

  // Handle scenario form submission
  const handleCreateScenario = () => {
    if (!scenarioForm.name.trim()) {
      alert('Please enter a scenario name');
      return;
    }

    createScenarioMutation.mutate(scenarioForm);
  };

  // Handle scenario execution
  const handleExecuteScenario = (scenarioId: string) => {
    executeScenarioMutation.mutate(scenarioId);
  };

  // Get status badge color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'running': return 'bg-blue-100 text-blue-800';
      case 'failed': return 'bg-red-100 text-red-800';
      case 'queued': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">Scenario Builder</h1>
        <Button onClick={() => setActiveTab('create')}>
          Create New Scenario
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="create">Create Scenario</TabsTrigger>
          <TabsTrigger value="manage">Manage Scenarios</TabsTrigger>
          <TabsTrigger value="runs">Scenario Runs</TabsTrigger>
        </TabsList>

        {/* Create Scenario Tab */}
        <TabsContent value="create" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Scenario Details</CardTitle>
              <CardDescription>Define your scenario parameters and assumptions</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="scenario-name">Scenario Name</Label>
                  <Input
                    id="scenario-name"
                    value={scenarioForm.name}
                    onChange={(e) => setScenarioForm(prev => ({ ...prev, name: e.target.value }))}
                    placeholder="e.g., High Demand Growth Scenario"
                  />
                </div>
                <div>
                  <Label htmlFor="scenario-category">Category</Label>
                  <Select
                    value={scenarioForm.category}
                    onValueChange={(value) => setScenarioForm(prev => ({ ...prev, category: value as any }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="demand">Demand</SelectItem>
                      <SelectItem value="supply">Supply</SelectItem>
                      <SelectItem value="policy">Policy</SelectItem>
                      <SelectItem value="economic">Economic</SelectItem>
                      <SelectItem value="weather">Weather</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div>
                <Label htmlFor="scenario-description">Description</Label>
                <Textarea
                  id="scenario-description"
                  value={scenarioForm.description}
                  onChange={(e) => setScenarioForm(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="Describe the scenario and its purpose..."
                  rows={3}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Start Date</Label>
                  <Input
                    type="date"
                    value={scenarioForm.time_horizon.start}
                    onChange={(e) => setScenarioForm(prev => ({
                      ...prev,
                      time_horizon: { ...prev.time_horizon, start: e.target.value }
                    }))}
                  />
                </div>
                <div>
                  <Label>End Date</Label>
                  <Input
                    type="date"
                    value={scenarioForm.time_horizon.end}
                    onChange={(e) => setScenarioForm(prev => ({
                      ...prev,
                      time_horizon: { ...prev.time_horizon, end: e.target.value }
                    }))}
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                Scenario Assumptions
                <Button onClick={addAssumption} size="sm">
                  <Plus className="w-4 h-4 mr-2" />
                  Add Assumption
                </Button>
              </CardTitle>
              <CardDescription>Define the key assumptions that will drive this scenario</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {scenarioForm.assumptions.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  No assumptions defined. Click "Add Assumption" to get started.
                </div>
              ) : (
                scenarioForm.assumptions.map((assumption) => (
                  <Card key={assumption.id} className="border-l-4 border-l-blue-500">
                    <CardContent className="pt-4">
                      <div className="flex justify-between items-start mb-3">
                        <div className="flex-1 grid grid-cols-2 gap-3">
                          <div>
                            <Label>Assumption Name</Label>
                            <Input
                              value={assumption.name}
                              onChange={(e) => updateAssumption(assumption.id, { name: e.target.value })}
                              placeholder="e.g., Load Growth Rate"
                            />
                          </div>
                          <div>
                            <Label>Type</Label>
                            <Select
                              value={assumption.type}
                              onValueChange={(value) => updateAssumption(assumption.id, { type: value as any })}
                            >
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="percentage">Percentage</SelectItem>
                                <SelectItem value="multiplier">Multiplier</SelectItem>
                                <SelectItem value="absolute">Absolute</SelectItem>
                                <SelectItem value="distribution">Distribution</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeAssumption(assumption.id)}
                          className="text-red-600 hover:text-red-800"
                        >
                          <Minus className="w-4 h-4" />
                        </Button>
                      </div>

                      <div className="grid grid-cols-3 gap-3">
                        <div>
                          <Label>Value</Label>
                          <Input
                            type="number"
                            step="0.01"
                            value={assumption.value}
                            onChange={(e) => updateAssumption(assumption.id, { value: parseFloat(e.target.value) || 0 })}
                          />
                        </div>

                        {assumption.type === 'distribution' && (
                          <>
                            <div>
                              <Label>Distribution</Label>
                              <Select
                                value={assumption.distribution || 'normal'}
                                onValueChange={(value) => updateAssumption(assumption.id, { distribution: value as any })}
                              >
                                <SelectTrigger>
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="normal">Normal</SelectItem>
                                  <SelectItem value="uniform">Uniform</SelectItem>
                                  <SelectItem value="beta">Beta</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>

                            {assumption.distribution === 'normal' && (
                              <div>
                                <Label>Std Dev</Label>
                                <Input
                                  type="number"
                                  step="0.01"
                                  value={assumption.std_dev || 0}
                                  onChange={(e) => updateAssumption(assumption.id, { std_dev: parseFloat(e.target.value) || 0 })}
                                />
                              </div>
                            )}

                            {assumption.distribution === 'uniform' && (
                              <>
                                <div>
                                  <Label>Min</Label>
                                  <Input
                                    type="number"
                                    step="0.01"
                                    value={assumption.min || 0}
                                    onChange={(e) => updateAssumption(assumption.id, { min: parseFloat(e.target.value) || 0 })}
                                  />
                                </div>
                                <div>
                                  <Label>Max</Label>
                                  <Input
                                    type="number"
                                    step="0.01"
                                    value={assumption.max || 0}
                                    onChange={(e) => updateAssumption(assumption.id, { max: parseFloat(e.target.value) || 0 })}
                                  />
                                </div>
                              </>
                            )}

                            {assumption.distribution === 'beta' && (
                              <>
                                <div>
                                  <Label>Alpha</Label>
                                  <Input
                                    type="number"
                                    step="0.1"
                                    value={assumption.alpha || 1}
                                    onChange={(e) => updateAssumption(assumption.id, { alpha: parseFloat(e.target.value) || 1 })}
                                  />
                                </div>
                                <div>
                                  <Label>Beta</Label>
                                  <Input
                                    type="number"
                                    step="0.1"
                                    value={assumption.beta || 1}
                                    onChange={(e) => updateAssumption(assumption.id, { beta: parseFloat(e.target.value) || 1 })}
                                  />
                                </div>
                              </>
                            )}
                          </>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Run Configuration</CardTitle>
              <CardDescription>Configure how the scenario should be executed</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Parallel Runs</Label>
                  <Input
                    type="number"
                    min="1"
                    max="1000"
                    value={scenarioForm.run_config.parallel_runs}
                    onChange={(e) => setScenarioForm(prev => ({
                      ...prev,
                      run_config: { ...prev.run_config, parallel_runs: parseInt(e.target.value) || 100 }
                    }))}
                  />
                </div>
                <div>
                  <Label>Output Granularity</Label>
                  <Select
                    value={scenarioForm.run_config.output_granularity}
                    onValueChange={(value) => setScenarioForm(prev => ({
                      ...prev,
                      run_config: { ...prev.run_config, output_granularity: value as any }
                    }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="hourly">Hourly</SelectItem>
                      <SelectItem value="daily">Daily</SelectItem>
                      <SelectItem value="monthly">Monthly</SelectItem>
                      <SelectItem value="quarterly">Quarterly</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="confidence-intervals"
                  checked={scenarioForm.run_config.include_confidence_intervals}
                  onChange={(e) => setScenarioForm(prev => ({
                    ...prev,
                    run_config: { ...prev.run_config, include_confidence_intervals: e.target.checked }
                  }))}
                />
                <Label htmlFor="confidence-intervals">Include Confidence Intervals</Label>
              </div>
            </CardContent>
          </Card>

          <div className="flex justify-end space-x-3">
            <Button variant="outline">Save Draft</Button>
            <Button onClick={handleCreateScenario} disabled={createScenarioMutation.isPending}>
              {createScenarioMutation.isPending ? 'Creating...' : 'Create Scenario'}
            </Button>
          </div>
        </TabsContent>

        {/* Manage Scenarios Tab */}
        <TabsContent value="manage" className="space-y-6">
          {scenariosLoading ? (
            <div className="text-center py-8">Loading scenarios...</div>
          ) : scenarios?.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              No scenarios created yet. Create your first scenario to get started.
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {scenarios?.map((scenario: any) => (
                <Card key={scenario.scenario_id}>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      {scenario.name}
                      <Badge className={getStatusColor(scenario.status)}>
                        {scenario.status}
                      </Badge>
                    </CardTitle>
                    <CardDescription>{scenario.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="text-sm text-gray-600">
                      <div>Category: {scenario.category}</div>
                      <div>Assumptions: {scenario.assumptions?.length || 0}</div>
                      <div>Created: {new Date(scenario.created_at).toLocaleDateString()}</div>
                    </div>

                    <div className="flex space-x-2">
                      <Button
                        size="sm"
                        onClick={() => handleExecuteScenario(scenario.scenario_id)}
                        disabled={executeScenarioMutation.isPending}
                      >
                        <Play className="w-4 h-4 mr-2" />
                        Run
                      </Button>
                      <Button variant="outline" size="sm">
                        Edit
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => deleteScenarioMutation.mutate(scenario.scenario_id)}
                        className="text-red-600 hover:text-red-800"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        {/* Scenario Runs Tab */}
        <TabsContent value="runs" className="space-y-6">
          {runsLoading ? (
            <div className="text-center py-8">Loading scenario runs...</div>
          ) : runs?.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              No scenario runs yet. Execute a scenario to see results here.
            </div>
          ) : (
            <div className="space-y-4">
              {runs?.map((run: ScenarioRun) => (
                <Card key={run.run_id}>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      Run {run.run_id}
                      <Badge className={getStatusColor(run.status)}>
                        {run.status}
                      </Badge>
                    </CardTitle>
                    <CardDescription>
                      Scenario: {run.scenario_id} • Started: {new Date(run.started_at).toLocaleString()}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {run.status === 'running' && (
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Progress</span>
                          <span>{run.progress}%</span>
                        </div>
                        <Progress value={run.progress} className="w-full" />
                      </div>
                    )}

                    {run.status === 'completed' && run.results && (
                      <div className="bg-green-50 p-3 rounded-lg">
                        <h4 className="font-medium text-green-800 mb-2">Results Summary</h4>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>Total Runs: {run.results.summary?.total_runs || 'N/A'}</div>
                          <div>Success Rate: {run.results.summary?.success_rate || 'N/A'}%</div>
                          <div>Mean Price Impact: ${run.results.summary?.mean_price_impact || 'N/A'}</div>
                          <div>Price Volatility: {run.results.summary?.price_volatility || 'N/A'}%</div>
                        </div>
                      </div>
                    )}

                    {run.status === 'failed' && (
                      <Alert>
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>
                          Scenario execution failed. Check logs for details.
                        </AlertDescription>
                      </Alert>
                    )}

                    <div className="flex justify-end space-x-2">
                      <Button variant="outline" size="sm">
                        View Details
                      </Button>
                      {run.status === 'completed' && (
                        <Button size="sm">
                          Download Results
                        </Button>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

