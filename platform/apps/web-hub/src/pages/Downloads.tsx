import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
// If additional UI elements (alert, progress, checkbox) are missing, ensure they exist in ui directory
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AlertCircle, Download, RefreshCw, Trash2, Plus } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Checkbox } from '@/components/ui/checkbox';
import { apiClient } from '@/services/api';

interface ExportRequest {
  id: string;
  name: string;
  format: 'csv' | 'parquet' | 'json' | 'xlsx';
  data_source: 'prices' | 'curves' | 'fundamentals' | 'backtests';
  instruments: string[];
  date_range: {
    start: string;
    end: string;
  };
  filters?: Record<string, any>;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  created_at: string;
  completed_at?: string;
  download_url?: string;
  file_size?: number;
  error_message?: string;
}

interface ExportJob {
  job_id: string;
  request_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  started_at: string;
  completed_at?: string;
  results?: {
    total_records: number;
    file_size: number;
    download_url: string;
  };
  error?: string;
}

export default function Downloads() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState('requests');

  // Form state for new export request
  const [exportForm, setExportForm] = useState({
    name: '',
    format: 'csv' as 'csv' | 'parquet' | 'json' | 'xlsx',
    data_source: 'prices' as 'prices' | 'curves' | 'fundamentals' | 'backtests',
    instruments: [] as string[],
    date_range: {
      start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      end: new Date().toISOString().split('T')[0]
    },
    filters: {} as Record<string, any>
  });

  // Query for export requests
  const { data: exportRequests, isLoading: requestsLoading } = useQuery({
    queryKey: ['export-requests'],
    queryFn: () => apiClient.get('/exports/requests').then(res => res.data.requests || [])
  });

  // Query for export jobs
  const { data: exportJobs, isLoading: jobsLoading } = useQuery({
    queryKey: ['export-jobs'],
    queryFn: () => apiClient.get('/exports/jobs').then(res => res.data.jobs || [])
  });

  // Mutations
  const createExportMutation = useMutation({
    mutationFn: (request: any) => apiClient.post('/exports/requests', request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['export-requests'] });
      setActiveTab('jobs');
    }
  });

  const deleteExportMutation = useMutation({
    mutationFn: (requestId: string) => apiClient.delete(`/exports/requests/${requestId}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['export-requests'] })
  });

  // Available instruments for selection
  const availableInstruments = [
    'MISO.HUB.INDIANA',
    'MISO.HUB.MICHIGAN_HUB',
    'PJM.HUB.WEST',
    'PJM.HUB.EAST',
    'CAISO.HUB.SP15',
    'CAISO.HUB.NP15',
    'ERCOT.HUB.NORTH',
    'NYISO.HUB.NYC'
  ];

  // Handle instrument selection
  const toggleInstrument = (instrument: string) => {
    setExportForm(prev => ({
      ...prev,
      instruments: prev.instruments.includes(instrument)
        ? prev.instruments.filter(i => i !== instrument)
        : [...prev.instruments, instrument]
    }));
  };

  // Handle export request submission
  const handleCreateExport = () => {
    if (!exportForm.name.trim()) {
      alert('Please enter an export name');
      return;
    }

    if (exportForm.instruments.length === 0) {
      alert('Please select at least one instrument');
      return;
    }

    createExportMutation.mutate({
      ...exportForm,
      filters: exportForm.filters,
      created_at: new Date().toISOString()
    });
  };

  // Get status badge color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'processing': return 'bg-blue-100 text-blue-800';
      case 'failed': return 'bg-red-100 text-red-800';
      case 'queued': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  // Format file size
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  // Handle download
  const handleDownload = (url: string, filename: string) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">Data Exports & Downloads</h1>
        <Button onClick={() => setActiveTab('create')}>
          <Plus className="w-4 h-4 mr-2" />
          New Export
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="create">Create Export</TabsTrigger>
          <TabsTrigger value="requests">Export Requests</TabsTrigger>
          <TabsTrigger value="jobs">Active Jobs</TabsTrigger>
        </TabsList>

        {/* Create Export Tab */}
        <TabsContent value="create" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Export Configuration</CardTitle>
              <CardDescription>Configure your data export request</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="export-name">Export Name</Label>
                  <Input
                    id="export-name"
                    value={exportForm.name}
                    onChange={(e) => setExportForm(prev => ({ ...prev, name: e.target.value }))}
                    placeholder="e.g., Monthly Price Data Export"
                  />
                </div>
                <div>
                  <Label htmlFor="export-format">Format</Label>
                  <Select
                    value={exportForm.format}
                    onValueChange={(value) => setExportForm(prev => ({ ...prev, format: value as any }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="csv">CSV</SelectItem>
                      <SelectItem value="parquet">Parquet</SelectItem>
                      <SelectItem value="json">JSON</SelectItem>
                      <SelectItem value="xlsx">Excel (XLSX)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div>
                <Label htmlFor="data-source">Data Source</Label>
                <Select
                  value={exportForm.data_source}
                  onValueChange={(value) => setExportForm(prev => ({ ...prev, data_source: value as any }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="prices">Price Data</SelectItem>
                    <SelectItem value="curves">Forward Curves</SelectItem>
                    <SelectItem value="fundamentals">Fundamentals</SelectItem>
                    <SelectItem value="backtests">Backtest Results</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Start Date</Label>
                  <Input
                    type="date"
                    value={exportForm.date_range.start}
                    onChange={(e) => setExportForm(prev => ({
                      ...prev,
                      date_range: { ...prev.date_range, start: e.target.value }
                    }))}
                  />
                </div>
                <div>
                  <Label>End Date</Label>
                  <Input
                    type="date"
                    value={exportForm.date_range.end}
                    onChange={(e) => setExportForm(prev => ({
                      ...prev,
                      date_range: { ...prev.date_range, end: e.target.value }
                    }))}
                  />
                </div>
              </div>

              <div>
                <Label>Instruments</Label>
                <div className="mt-2 max-h-48 overflow-y-auto border rounded-lg p-3 bg-gray-50">
                  <div className="grid grid-cols-2 gap-2">
                    {availableInstruments.map((instrument) => (
                      <div key={instrument} className="flex items-center space-x-2">
                        <Checkbox
                          id={instrument}
                          checked={exportForm.instruments.includes(instrument)}
                          onCheckedChange={() => toggleInstrument(instrument)}
                        />
                        <Label htmlFor={instrument} className="text-sm">
                          {instrument}
                        </Label>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="mt-2 text-sm text-gray-600">
                  Selected: {exportForm.instruments.length} instruments
                </div>
              </div>

              <div className="flex justify-end space-x-3">
                <Button variant="outline">Save Template</Button>
                <Button onClick={handleCreateExport} disabled={createExportMutation.isPending}>
                  {createExportMutation.isPending ? 'Creating...' : 'Create Export Request'}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Export Requests Tab */}
        <TabsContent value="requests" className="space-y-6">
          {requestsLoading ? (
            <div className="text-center py-8">Loading export requests...</div>
          ) : exportRequests?.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              No export requests yet. Create your first export to get started.
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {exportRequests?.map((request: ExportRequest) => (
                <Card key={request.id}>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      {request.name}
                      <Badge className={getStatusColor(request.status)}>
                        {request.status}
                      </Badge>
                    </CardTitle>
                    <CardDescription>
                      {request.data_source} • {request.format.toUpperCase()}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="text-sm text-gray-600">
                      <div>Instruments: {request.instruments.length}</div>
                      <div>Created: {new Date(request.created_at).toLocaleDateString()}</div>
                      {request.file_size && (
                        <div>File Size: {formatFileSize(request.file_size)}</div>
                      )}
                    </div>

                    {request.status === 'processing' && (
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Progress</span>
                          <span>{request.progress}%</span>
                        </div>
                        <Progress value={request.progress} className="w-full" />
                      </div>
                    )}

                    {request.status === 'completed' && request.download_url && (
                      <Button
                        onClick={() => handleDownload(request.download_url!, `${request.name}.${request.format}`)}
                        className="w-full"
                      >
                        <Download className="w-4 h-4 mr-2" />
                        Download
                      </Button>
                    )}

                    {request.status === 'failed' && request.error_message && (
                      <Alert>
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription className="text-sm">
                          {request.error_message}
                        </AlertDescription>
                      </Alert>
                    )}

                    <div className="flex space-x-2">
                      <Button variant="outline" size="sm">
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Retry
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => deleteExportMutation.mutate(request.id)}
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

        {/* Active Jobs Tab */}
        <TabsContent value="jobs" className="space-y-6">
          {jobsLoading ? (
            <div className="text-center py-8">Loading export jobs...</div>
          ) : exportJobs?.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              No active export jobs. Create an export request to see jobs here.
            </div>
          ) : (
            <div className="space-y-4">
              {exportJobs?.map((job: ExportJob) => (
                <Card key={job.job_id}>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      Job {job.job_id}
                      <Badge className={getStatusColor(job.status)}>
                        {job.status}
                      </Badge>
                    </CardTitle>
                    <CardDescription>
                      Started: {new Date(job.started_at).toLocaleString()}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {job.status === 'processing' && (
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Progress</span>
                          <span>{job.progress}%</span>
                        </div>
                        <Progress value={job.progress} className="w-full" />
                      </div>
                    )}

                    {job.status === 'completed' && job.results && (
                      <div className="bg-green-50 p-3 rounded-lg">
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>Total Records: {job.results.total_records.toLocaleString()}</div>
                          <div>File Size: {formatFileSize(job.results.file_size)}</div>
                        </div>
                        <Button
                          onClick={() => handleDownload(job.results!.download_url, `export-${job.job_id}.${exportForm.format}`)}
                          className="w-full mt-2"
                        >
                          <Download className="w-4 h-4 mr-2" />
                          Download File
                        </Button>
                      </div>
                    )}

                    {job.status === 'failed' && job.error && (
                      <Alert>
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription className="text-sm">
                          {job.error}
                        </AlertDescription>
                      </Alert>
                    )}

                    <div className="flex justify-end">
                      <Button variant="outline" size="sm">
                        View Details
                      </Button>
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

