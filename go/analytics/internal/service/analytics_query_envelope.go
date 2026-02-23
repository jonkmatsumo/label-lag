package service

import (
	"fmt"
	"time"

	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"google.golang.org/protobuf/types/known/timestamppb"
)

const analyticsDateLayout = "2006-01-02"

func parseAnalyticsEnvelopeTime(raw string) (*timestamppb.Timestamp, error) {
	if raw == "" {
		return nil, nil
	}
	if parsed, err := time.Parse(time.RFC3339, raw); err == nil {
		return timestamppb.New(parsed), nil
	}
	if parsed, err := time.Parse(analyticsDateLayout, raw); err == nil {
		return timestamppb.New(parsed), nil
	}
	return nil, fmt.Errorf("invalid time format: %s", raw)
}

func mergeWindowFromEnvelope(
	legacyStart *timestamppb.Timestamp,
	legacyEnd *timestamppb.Timestamp,
	query *pb.AnalyticsQueryEnvelope,
) (*timestamppb.Timestamp, *timestamppb.Timestamp, error) {
	start := legacyStart
	end := legacyEnd
	var err error

	if start == nil && query != nil && query.GetStartTime() != "" {
		start, err = parseAnalyticsEnvelopeTime(query.GetStartTime())
		if err != nil {
			return nil, nil, err
		}
	}
	if end == nil && query != nil && query.GetEndTime() != "" {
		end, err = parseAnalyticsEnvelopeTime(query.GetEndTime())
		if err != nil {
			return nil, nil, err
		}
	}
	return start, end, nil
}

func mergeGranularityFromEnvelope(legacy string, query *pb.AnalyticsQueryEnvelope, fallback string) string {
	if legacy != "" {
		return legacy
	}
	if query != nil && query.GetGranularity() != "" {
		return query.GetGranularity()
	}
	return fallback
}
