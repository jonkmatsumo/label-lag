package main

import (
	"context"
	"database/sql"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/jonkmatsumo/label-lag/go/analytics/internal/config"
	_ "github.com/lib/pq"
)

func main() {
	startStr := flag.String("start", "", "Start date (YYYY-MM-DD or RFC3339)")
	endStr := flag.String("end", "", "End date (YYYY-MM-DD or RFC3339)")
	granularity := flag.String("granularity", "day", "Granularity (day or hour)")
	limitMismatches := flag.Int("limit-mismatches", 10, "Limit number of mismatches to report")
	flag.Parse()

	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	slog.SetDefault(logger)

	dbURL, err := config.ResolveDatabaseURL(os.Getenv)
	if err != nil {
		slog.Error("failed to resolve database url", "error", err)
		os.Exit(1)
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		slog.Error("failed to connect to database", "error", err)
		os.Exit(1)
	}
	defer db.Close()

	ctx := context.Background()

	startTime, ok := parseTime(*startStr)
	if *startStr != "" && !ok {
		slog.Error("invalid start time", "val", *startStr)
		os.Exit(1)
	}
	endTime, ok := parseTime(*endStr)
	if *endStr != "" && !ok {
		slog.Error("invalid end time", "val", *endStr)
		os.Exit(1)
	}

	if *granularity != "day" && *granularity != "hour" {
		slog.Error("invalid granularity", "val", *granularity)
		os.Exit(1)
	}

	slog.Info("starting reconciliation", "granularity", *granularity, "start", startTime, "end", endTime)

	var query string
	if *granularity == "day" {
		query = buildDailyReconcileQuery(startTime, endTime)
	} else {
		query = buildHourlyReconcileQuery(startTime, endTime)
	}

	rows, err := db.QueryContext(ctx, query)
	if err != nil {
		slog.Error("failed to run reconciliation query", "error", err)
		os.Exit(1)
	}
	defer rows.Close()

	mismatches := 0
	for rows.Next() {
		var period string
		var rawCount, aggCount, rawAlerts, aggAlerts int64
		var rawSumScore, aggSumScore int64

		if err := rows.Scan(&period, &rawCount, &aggCount, &rawAlerts, &aggAlerts, &rawSumScore, &aggSumScore); err != nil {
			slog.Error("failed to scan row", "error", err)
			os.Exit(1)
		}

		if rawCount != aggCount || rawAlerts != aggAlerts || rawSumScore != aggSumScore {
			mismatches++
			if mismatches <= *limitMismatches {
				slog.Warn("mismatch detected",
					"period", period,
					"raw_count", rawCount, "agg_count", aggCount,
					"raw_alerts", rawAlerts, "agg_alerts", aggAlerts,
					"raw_sum_score", rawSumScore, "agg_sum_score", aggSumScore)
			}
		}
	}

	if mismatches > 0 {
		slog.Error("reconciliation failed", "total_mismatches", mismatches)
		os.Exit(1)
	}

	slog.Info("reconciliation complete: no mismatches found")
}

func parseTime(s string) (time.Time, bool) {
	if s == "" {
		return time.Time{}, true
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t, true
	}
	if t, err := time.Parse("2006-01-02", s); err == nil {
		return t, true
	}
	return time.Time{}, false
}

func buildDailyReconcileQuery(start, end time.Time) string {
	where := "WHERE 1=1"
	if !start.IsZero() {
		where += fmt.Sprintf(" AND ts >= '%s'", start.Format(time.RFC3339))
	}
	if !end.IsZero() {
		where += fmt.Sprintf(" AND ts <= '%s'", end.Format(time.RFC3339))
	}

	return fmt.Sprintf(`
		WITH raw_daily AS (
			SELECT
				DATE(ts) as d,
				COUNT(*) as total_decisions,
				SUM(CASE WHEN decision IN ('REVIEW', 'REJECT') THEN 1 ELSE 0 END) as total_alerts,
				SUM(final_score) as sum_score
			FROM inference_events
			%s
			GROUP BY 1
		)
		SELECT
			rd.d::text,
			rd.total_decisions as raw_count,
			COALESCE(ad.total_decisions, 0) as agg_count,
			rd.total_alerts as raw_alerts,
			COALESCE(ad.total_alerts, 0) as agg_alerts,
			rd.sum_score as raw_sum_score,
			COALESCE(ad.sum_score, 0) as agg_sum_score
		FROM raw_daily rd
		FULL OUTER JOIN aggregates_daily ad ON rd.d = ad.date
		WHERE rd.total_decisions != COALESCE(ad.total_decisions, 0)
		   OR rd.total_alerts != COALESCE(ad.total_alerts, 0)
		   OR rd.sum_score != COALESCE(ad.sum_score, 0)
	`, where)
}

func buildHourlyReconcileQuery(start, end time.Time) string {
	where := "WHERE 1=1"
	if !start.IsZero() {
		where += fmt.Sprintf(" AND ts >= '%s'", start.Format(time.RFC3339))
	}
	if !end.IsZero() {
		where += fmt.Sprintf(" AND ts <= '%s'", end.Format(time.RFC3339))
	}

	return fmt.Sprintf(`
		WITH raw_hourly AS (
			SELECT
				date_trunc('hour', ts) as h,
				COUNT(*) as total_decisions,
				SUM(CASE WHEN decision IN ('REVIEW', 'REJECT') THEN 1 ELSE 0 END) as total_alerts,
				SUM(final_score) as sum_score
			FROM inference_events
			%s
			GROUP BY 1
		)
		SELECT
			rh.h::text,
			rh.total_decisions as raw_count,
			COALESCE(ah.total_decisions, 0) as agg_count,
			rh.total_alerts as raw_alerts,
			COALESCE(ah.total_alerts, 0) as agg_alerts,
			rh.sum_score as raw_sum_score,
			COALESCE(ah.sum_score, 0) as agg_sum_score
		FROM raw_hourly rh
		FULL OUTER JOIN aggregates_hourly ah ON rh.h = ah.hour
		WHERE rh.total_decisions != COALESCE(ah.total_decisions, 0)
		   OR rh.total_alerts != COALESCE(ah.total_alerts, 0)
		   OR rh.sum_score != COALESCE(ah.sum_score, 0)
	`, where)
}
