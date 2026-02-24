package store

import (
	"time"

	"google.golang.org/protobuf/types/known/timestamppb"
)

const (
	MaxPointsHourly = 1000
	MaxPointsDaily  = 2000
)

type timeSeriesPlan struct {
	effectiveGranularity string
	pointCap             int
	partial              bool
}

func buildTimeSeriesPlan(
	requestedGranularity string,
	start *timestamppb.Timestamp,
	end *timestamppb.Timestamp,
) timeSeriesPlan {
	granularity := requestedGranularity
	if granularity != "hour" {
		granularity = "day"
	}

	partial := false
	if granularity == "hour" && estimateBucketCount(start, end, "hour") > MaxPointsHourly {
		// Deterministically downsample oversized hourly windows to day buckets.
		granularity = "day"
		partial = true
	}

	pointCap := MaxPointsDaily
	if granularity == "hour" {
		pointCap = MaxPointsHourly
	}
	if estimateBucketCount(start, end, granularity) > pointCap {
		partial = true
	}

	return timeSeriesPlan{
		effectiveGranularity: granularity,
		pointCap:             pointCap,
		partial:              partial,
	}
}

func estimateBucketCount(
	start *timestamppb.Timestamp,
	end *timestamppb.Timestamp,
	granularity string,
) int {
	if start == nil || end == nil {
		return 0
	}

	startTime := start.AsTime().UTC()
	endTime := end.AsTime().UTC()
	if endTime.Before(startTime) {
		return 0
	}

	switch granularity {
	case "hour":
		startHour := startTime.Truncate(time.Hour)
		endHour := endTime.Truncate(time.Hour)
		return int(endHour.Sub(startHour)/time.Hour) + 1
	default:
		startDay := time.Date(startTime.Year(), startTime.Month(), startTime.Day(), 0, 0, 0, 0, time.UTC)
		endDay := time.Date(endTime.Year(), endTime.Month(), endTime.Day(), 0, 0, 0, 0, time.UTC)
		return int(endDay.Sub(startDay)/(24*time.Hour)) + 1
	}
}
