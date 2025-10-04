{{/*
Expand the name of the chart.
*/}}
{{- define "market-intelligence.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "market-intelligence.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "market-intelligence.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "market-intelligence.labels" -}}
helm.sh/chart: {{ include "market-intelligence.chart" . }}
{{ include "market-intelligence.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: market-intelligence
{{- end }}

{{/*
Selector labels
*/}}
{{- define "market-intelligence.selectorLabels" -}}
app.kubernetes.io/name: {{ include "market-intelligence.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "market-intelligence.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "market-intelligence.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Blue-green deployment helpers
*/}}
{{- define "market-intelligence.blue.fullname" -}}
{{- printf "%s-blue" (include "market-intelligence.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "market-intelligence.green.fullname" -}}
{{- printf "%s-green" (include "market-intelligence.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "market-intelligence.activeColor" -}}
{{- if .Values.blueGreen.enabled }}
{{- .Values.blueGreen.primaryColor }}
{{- else }}
{{- "blue" }}
{{- end }}
{{- end }}

{{/*
Generate full image name
*/}}
{{- define "market-intelligence.image" -}}
{{- $registry := default .Values.global.imageRegistry .Values.image.registry -}}
{{- $repository := .repository -}}
{{- $tag := default .Chart.AppVersion .tag -}}
{{- if $registry -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- else -}}
{{- printf "%s:%s" $repository $tag -}}
{{- end -}}
{{- end -}}
