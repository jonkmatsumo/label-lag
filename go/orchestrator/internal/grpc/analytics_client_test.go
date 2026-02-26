package grpc

import (
	"context"
	"testing"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/mock"
	"google.golang.org/grpc"
)

type mockAnalyticsService struct {
	mock.Mock
}

func (m *mockAnalyticsService) GetDailyStats(ctx context.Context, in *crudv1.GetDailyStatsRequest, opts ...grpc.CallOption) (*crudv1.GetDailyStatsResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetTransactionDetails(ctx context.Context, in *crudv1.GetTransactionDetailsRequest, opts ...grpc.CallOption) (*crudv1.GetTransactionDetailsResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) SearchTransactions(ctx context.Context, in *crudv1.SearchTransactionsRequest, opts ...grpc.CallOption) (*crudv1.SearchTransactionsResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetRecentAlerts(ctx context.Context, in *crudv1.GetRecentAlertsRequest, opts ...grpc.CallOption) (*crudv1.GetRecentAlertsResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetOverviewMetrics(ctx context.Context, in *crudv1.GetOverviewMetricsRequest, opts ...grpc.CallOption) (*crudv1.GetOverviewMetricsResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetFeatureSample(ctx context.Context, in *crudv1.GetFeatureSampleRequest, opts ...grpc.CallOption) (*crudv1.GetFeatureSampleResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetTrainingData(ctx context.Context, in *crudv1.GetTrainingDataRequest, opts ...grpc.CallOption) (*crudv1.GetTrainingDataResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetBacktestFeatures(ctx context.Context, in *crudv1.GetBacktestFeaturesRequest, opts ...grpc.CallOption) (*crudv1.GetBacktestFeaturesResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) SaveBacktestResult(ctx context.Context, in *crudv1.SaveBacktestResultRequest, opts ...grpc.CallOption) (*crudv1.SaveBacktestResultResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) ListBacktestResults(ctx context.Context, in *crudv1.ListBacktestResultsRequest, opts ...grpc.CallOption) (*crudv1.ListBacktestResultsResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetBacktestResult(ctx context.Context, in *crudv1.GetBacktestResultRequest, opts ...grpc.CallOption) (*crudv1.GetBacktestResultResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetDriftWindow(ctx context.Context, in *crudv1.GetDriftWindowRequest, opts ...grpc.CallOption) (*crudv1.GetDriftWindowResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) StoreGeneratedData(ctx context.Context, in *crudv1.StoreGeneratedDataRequest, opts ...grpc.CallOption) (*crudv1.StoreGeneratedDataResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) ClearAllData(ctx context.Context, in *crudv1.ClearAllDataRequest, opts ...grpc.CallOption) (*crudv1.ClearAllDataResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) MaterializeFeatures(ctx context.Context, in *crudv1.MaterializeFeaturesRequest, opts ...grpc.CallOption) (*crudv1.MaterializeFeaturesResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GenerateData(ctx context.Context, in *crudv1.GenerateDataRequest, opts ...grpc.CallOption) (*crudv1.GenerateDataResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetInferenceScores(ctx context.Context, in *crudv1.GetInferenceScoresRequest, opts ...grpc.CallOption) (*crudv1.GetInferenceScoresResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) SaveRule(ctx context.Context, in *crudv1.SaveRuleRequest, opts ...grpc.CallOption) (*crudv1.SaveRuleResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetRule(ctx context.Context, in *crudv1.GetRuleRequest, opts ...grpc.CallOption) (*crudv1.GetRuleResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) ListRules(ctx context.Context, in *crudv1.ListRulesRequest, opts ...grpc.CallOption) (*crudv1.ListRulesResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) DeleteRule(ctx context.Context, in *crudv1.DeleteRuleRequest, opts ...grpc.CallOption) (*crudv1.DeleteRuleResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) ListRuleVersions(ctx context.Context, in *crudv1.ListRuleVersionsRequest, opts ...grpc.CallOption) (*crudv1.ListRuleVersionsResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetRuleVersion(ctx context.Context, in *crudv1.GetRuleVersionRequest, opts ...grpc.CallOption) (*crudv1.GetRuleVersionResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) PublishRuleVersion(ctx context.Context, in *crudv1.PublishRuleVersionRequest, opts ...grpc.CallOption) (*crudv1.PublishRuleVersionResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetRuleReadiness(ctx context.Context, in *crudv1.GetRuleReadinessRequest, opts ...grpc.CallOption) (*crudv1.GetRuleReadinessResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) DiffRuleVersions(ctx context.Context, in *crudv1.DiffRuleVersionsRequest, opts ...grpc.CallOption) (*crudv1.DiffRuleVersionsResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetRuleStats(ctx context.Context, in *crudv1.GetRuleStatsRequest, opts ...grpc.CallOption) (*crudv1.GetRuleStatsResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetAttribution(ctx context.Context, in *crudv1.GetAttributionRequest, opts ...grpc.CallOption) (*crudv1.GetAttributionResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetLatestUserFeatures(ctx context.Context, in *crudv1.GetLatestUserFeaturesRequest, opts ...grpc.CallOption) (*crudv1.GetLatestUserFeaturesResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) BatchGetLatestUserFeatures(ctx context.Context, in *crudv1.BatchGetLatestUserFeaturesRequest, opts ...grpc.CallOption) (*crudv1.BatchGetLatestUserFeaturesResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) CompareBacktests(ctx context.Context, in *crudv1.CompareBacktestsRequest, opts ...grpc.CallOption) (*crudv1.CompareBacktestsResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetShadowComparison(ctx context.Context, in *crudv1.GetShadowComparisonRequest, opts ...grpc.CallOption) (*crudv1.GetShadowComparisonResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetDatasetProfile(ctx context.Context, in *crudv1.GetDatasetProfileRequest, opts ...grpc.CallOption) (*crudv1.GetDatasetProfileResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) ListDecisions(ctx context.Context, in *crudv1.ListDecisionsRequest, opts ...grpc.CallOption) (*crudv1.ListDecisionsResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetDecision(ctx context.Context, in *crudv1.GetDecisionRequest, opts ...grpc.CallOption) (*crudv1.GetDecisionResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetDecisionTrace(ctx context.Context, in *crudv1.GetDecisionTraceRequest, opts ...grpc.CallOption) (*crudv1.GetDecisionTraceResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetRuleImpact(ctx context.Context, in *crudv1.GetRuleImpactRequest, opts ...grpc.CallOption) (*crudv1.GetRuleImpactResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetKpis(ctx context.Context, in *crudv1.GetKpisRequest, opts ...grpc.CallOption) (*crudv1.GetKpisResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetVolumeSeries(ctx context.Context, in *crudv1.GetVolumeSeriesRequest, opts ...grpc.CallOption) (*crudv1.GetVolumeSeriesResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetConfusionMatrix(ctx context.Context, in *crudv1.GetConfusionMatrixRequest, opts ...grpc.CallOption) (*crudv1.GetConfusionMatrixResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) ListJobs(ctx context.Context, in *crudv1.ListJobsRequest, opts ...grpc.CallOption) (*crudv1.ListJobsResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetJob(ctx context.Context, in *crudv1.GetJobRequest, opts ...grpc.CallOption) (*crudv1.GetJobResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetJobEvents(ctx context.Context, in *crudv1.GetJobEventsRequest, opts ...grpc.CallOption) (*crudv1.GetJobEventsResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetJobSummary(ctx context.Context, in *crudv1.GetJobSummaryRequest, opts ...grpc.CallOption) (*crudv1.GetJobSummaryResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) CancelJob(ctx context.Context, in *crudv1.CancelJobRequest, opts ...grpc.CallOption) (*crudv1.CancelJobResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) RetryJob(ctx context.Context, in *crudv1.RetryJobRequest, opts ...grpc.CallOption) (*crudv1.RetryJobResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetDatasetSummary(ctx context.Context, in *crudv1.GetDatasetSummaryRequest, opts ...grpc.CallOption) (*crudv1.GetDatasetSummaryResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) ListDatasetProfiles(ctx context.Context, in *crudv1.ListDatasetProfilesRequest, opts ...grpc.CallOption) (*crudv1.ListDatasetProfilesResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetLatestDatasetProfile(ctx context.Context, in *crudv1.GetLatestDatasetProfileRequest, opts ...grpc.CallOption) (*crudv1.GetLatestDatasetProfileResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) CompareDatasetProfiles(ctx context.Context, in *crudv1.CompareDatasetProfilesRequest, opts ...grpc.CallOption) (*crudv1.CompareDatasetProfilesResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) ListTrainingRuns(ctx context.Context, in *crudv1.ListTrainingRunsRequest, opts ...grpc.CallOption) (*crudv1.ListTrainingRunsResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetTrainingRun(ctx context.Context, in *crudv1.GetTrainingRunRequest, opts ...grpc.CallOption) (*crudv1.GetTrainingRunResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) ListModelVersions(ctx context.Context, in *crudv1.ListModelVersionsRequest, opts ...grpc.CallOption) (*crudv1.ListModelVersionsResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) GetMetricSeries(ctx context.Context, in *crudv1.GetMetricSeriesRequest, opts ...grpc.CallOption) (*crudv1.GetMetricSeriesResponse, error) {
	return nil, nil
}
func (m *mockAnalyticsService) ReportTrainingRun(ctx context.Context, in *crudv1.ReportTrainingRunRequest, opts ...grpc.CallOption) (*crudv1.ReportTrainingRunResponse, error) {
	return nil, nil
}

func (m *mockAnalyticsService) LogInferenceEvent(ctx context.Context, req *crudv1.LogInferenceEventRequest, opts ...grpc.CallOption) (*crudv1.LogInferenceEventResponse, error) {
	args := m.Called(mock.Anything, req)
	return args.Get(0).(*crudv1.LogInferenceEventResponse), args.Error(1)
}

func TestAnalyticsClient_LogInferenceEvent_Async(t *testing.T) {
	mockStub := new(mockAnalyticsService)
	client := &AnalyticsClient{
		timeout:  time.Second,
		stub:     mockStub,
		logQueue: make(chan *logQueueItem, 2),
		stop:     make(chan struct{}),
	}
	client.startWorker()
	defer client.Close()

	req := &crudv1.LogInferenceEventRequest{
		Event: &crudv1.InferenceEvent{
			RequestId: "test-id",
		},
	}
	mockStub.On("LogInferenceEvent", mock.Anything, req).Return(&crudv1.LogInferenceEventResponse{Success: true}, nil).Once()

	resp, err := client.LogInferenceEvent(context.Background(), req)
	if err != nil {
		t.Fatalf("Log failed: %v", err)
	}
	if !resp.Success {
		t.Error("expected success")
	}

	// Give worker time to process
	time.Sleep(100 * time.Millisecond)
	mockStub.AssertExpectations(t)
}

func TestAnalyticsClient_LogInferenceEvent_Drop(t *testing.T) {
	client := &AnalyticsClient{
		logQueue: make(chan *logQueueItem, 1),
		stop:     make(chan struct{}),
	}
	// Don't start worker to simulate full queue

	req := &crudv1.LogInferenceEventRequest{
		Event: &crudv1.InferenceEvent{
			RequestId: "drop-me",
		},
	}

	// Fill queue
	client.logQueue <- &logQueueItem{req: req, enqueuedAt: time.Now()}

	// This should drop
	resp, err := client.LogInferenceEvent(context.Background(), req)
	if err != nil {
		t.Fatalf("Log failed: %v", err)
	}
	if !resp.Success {
		t.Error("expected success even on drop")
	}
}

func TestAnalyticsClient_LogInferenceEvent_TracksEnqueueAttempts(t *testing.T) {
	analyticsLogQueueEnqueueAttempts.Reset()
	LogEventsDropped.Reset()

	client := &AnalyticsClient{
		logQueue: make(chan *logQueueItem, 1),
		stop:     make(chan struct{}),
	}
	req := &crudv1.LogInferenceEventRequest{
		Event: &crudv1.InferenceEvent{RequestId: "attempts"},
	}

	_, err := client.LogInferenceEvent(context.Background(), req)
	if err != nil {
		t.Fatalf("first enqueue failed: %v", err)
	}
	_, err = client.LogInferenceEvent(context.Background(), req)
	if err != nil {
		t.Fatalf("second enqueue failed: %v", err)
	}

	if got := testutil.ToFloat64(analyticsLogQueueEnqueueAttempts.WithLabelValues("analytics")); got != 2 {
		t.Fatalf("expected 2 enqueue attempts, got %v", got)
	}
	if got := testutil.ToFloat64(LogEventsDropped.WithLabelValues("analytics", "full")); got != 1 {
		t.Fatalf("expected 1 dropped-full event, got %v", got)
	}
}

func TestAnalyticsClient_QueueMetricsUseBoundedLabels(t *testing.T) {
	analyticsLogQueueEnqueueAttempts.Reset()
	LogEventsDropped.Reset()
	analyticsLogQueueDrainLatency.Reset()

	client := &AnalyticsClient{
		logQueue: make(chan *logQueueItem, 1),
		stop:     make(chan struct{}),
	}
	req := &crudv1.LogInferenceEventRequest{
		Event: &crudv1.InferenceEvent{RequestId: "bounded-labels"},
	}

	// First request enqueues, second request drops (queue full).
	if _, err := client.LogInferenceEvent(context.Background(), req); err != nil {
		t.Fatalf("first enqueue failed: %v", err)
	}
	if _, err := client.LogInferenceEvent(context.Background(), req); err != nil {
		t.Fatalf("second enqueue failed: %v", err)
	}

	attemptCounter, err := analyticsLogQueueEnqueueAttempts.GetMetricWithLabelValues("analytics")
	if err != nil {
		t.Fatalf("failed to read enqueue attempts metric: %v", err)
	}
	attemptMetric := &dto.Metric{}
	if err := attemptCounter.Write(attemptMetric); err != nil {
		t.Fatalf("failed to serialize enqueue attempts metric: %v", err)
	}
	assertMetricLabels(t, attemptMetric, map[string]string{
		"queue": "analytics",
	})

	dropCounter, err := LogEventsDropped.GetMetricWithLabelValues("analytics", "full")
	if err != nil {
		t.Fatalf("failed to read dropped metric: %v", err)
	}
	dropMetric := &dto.Metric{}
	if err := dropCounter.Write(dropMetric); err != nil {
		t.Fatalf("failed to serialize dropped metric: %v", err)
	}
	assertMetricLabels(t, dropMetric, map[string]string{
		"queue":  "analytics",
		"reason": "full",
	})

	drainObserver, err := analyticsLogQueueDrainLatency.GetMetricWithLabelValues("analytics")
	if err != nil {
		t.Fatalf("failed to read drain latency metric: %v", err)
	}
	drainHistogram, ok := drainObserver.(prometheus.Histogram)
	if !ok {
		t.Fatalf("drain latency metric is not a histogram")
	}
	drainMetric := &dto.Metric{}
	if err := drainHistogram.Write(drainMetric); err != nil {
		t.Fatalf("failed to serialize drain latency metric: %v", err)
	}
	assertMetricLabels(t, drainMetric, map[string]string{
		"queue": "analytics",
	})
}

func TestAnalyticsClient_WorkerRecordsDrainLatency(t *testing.T) {
	analyticsLogQueueDrainLatency.Reset()

	mockStub := new(mockAnalyticsService)
	client := &AnalyticsClient{
		timeout:  time.Second,
		stub:     mockStub,
		logQueue: make(chan *logQueueItem, 2),
		stop:     make(chan struct{}),
	}
	client.startWorker()
	defer client.Close()

	req := &crudv1.LogInferenceEventRequest{
		Event: &crudv1.InferenceEvent{
			RequestId: "latency-metric",
		},
	}
	mockStub.On("LogInferenceEvent", mock.Anything, req).Return(&crudv1.LogInferenceEventResponse{Success: true}, nil).Once()

	_, err := client.LogInferenceEvent(context.Background(), req)
	if err != nil {
		t.Fatalf("Log failed: %v", err)
	}
	time.Sleep(100 * time.Millisecond)

	observer, err := analyticsLogQueueDrainLatency.GetMetricWithLabelValues("analytics")
	if err != nil {
		t.Fatalf("failed to get drain latency observer: %v", err)
	}
	histogram, ok := observer.(prometheus.Histogram)
	if !ok {
		t.Fatalf("drain latency metric is not a histogram")
	}
	metric := &dto.Metric{}
	if err := histogram.Write(metric); err != nil {
		t.Fatalf("failed to read drain latency metric: %v", err)
	}
	if metric.GetHistogram().GetSampleCount() < 1 {
		t.Fatalf("expected drain latency sample count >= 1, got %d", metric.GetHistogram().GetSampleCount())
	}
	mockStub.AssertExpectations(t)
}

func assertMetricLabels(t *testing.T, metric *dto.Metric, expected map[string]string) {
	t.Helper()
	if len(metric.GetLabel()) != len(expected) {
		t.Fatalf("expected %d labels, got %d", len(expected), len(metric.GetLabel()))
	}
	for _, label := range metric.GetLabel() {
		want, ok := expected[label.GetName()]
		if !ok {
			t.Fatalf("unexpected label %q", label.GetName())
		}
		if label.GetValue() != want {
			t.Fatalf("label %q: got %q want %q", label.GetName(), label.GetValue(), want)
		}
	}
}
