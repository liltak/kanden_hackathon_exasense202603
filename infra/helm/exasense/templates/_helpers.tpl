{{/*
Common labels
*/}}
{{- define "exasense.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "exasense.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Full name with release
*/}}
{{- define "exasense.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Namespace
*/}}
{{- define "exasense.namespace" -}}
{{ .Values.global.namespace | default .Release.Namespace }}
{{- end }}

{{/*
Database URL
*/}}
{{- define "exasense.databaseUrl" -}}
postgresql+asyncpg://{{ .Values.postgres.auth.username }}:{{ .Values.postgres.auth.password }}@{{ include "exasense.fullname" . }}-postgres:5432/{{ .Values.postgres.auth.database }}
{{- end }}

{{/*
Redis URL
*/}}
{{- define "exasense.redisUrl" -}}
redis://{{ include "exasense.fullname" . }}-redis:6379/0
{{- end }}
