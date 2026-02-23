package httpserver

import (
	"fmt"
	"net/url"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"google.golang.org/protobuf/types/known/timestamppb"
)

const analyticsDateOnlyLayout = "2006-01-02"

func firstNonEmptyQuery(values url.Values, keys ...string) string {
	for _, key := range keys {
		if value := values.Get(key); value != "" {
			return value
		}
	}
	return ""
}

func parseAnalyticsTime(raw string) (*timestamppb.Timestamp, error) {
	if raw == "" {
		return nil, nil
	}
	if parsed, err := time.Parse(time.RFC3339, raw); err == nil {
		return timestamppb.New(parsed), nil
	}
	if parsed, err := time.Parse(analyticsDateOnlyLayout, raw); err == nil {
		return timestamppb.New(parsed), nil
	}
	return nil, fmt.Errorf("invalid analytics timestamp: %s", raw)
}

func parseAnalyticsTimeQuery(values url.Values, keys ...string) (string, *timestamppb.Timestamp, error) {
	raw := firstNonEmptyQuery(values, keys...)
	parsed, err := parseAnalyticsTime(raw)
	if err != nil {
		return raw, nil, err
	}
	return raw, parsed, nil
}

func buildAnalyticsQueryEnvelope(startRaw, endRaw, granularity string) *crudv1.AnalyticsQueryEnvelope {
	if startRaw == "" && endRaw == "" && granularity == "" {
		return nil
	}
	return &crudv1.AnalyticsQueryEnvelope{
		StartTime:   startRaw,
		EndTime:     endRaw,
		Granularity: granularity,
	}
}
