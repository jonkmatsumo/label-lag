package service

import (
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func mergeWindowFromEnvelope(
	legacyStart *timestamppb.Timestamp,
	legacyEnd *timestamppb.Timestamp,
	query *pb.AnalyticsQueryEnvelope,
) (*timestamppb.Timestamp, *timestamppb.Timestamp, error) {
	start := legacyStart
	end := legacyEnd

	if start == nil && query != nil && query.GetStartTime() != nil {
		start = query.GetStartTime()
	}
	if end == nil && query != nil && query.GetEndTime() != nil {
		end = query.GetEndTime()
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
