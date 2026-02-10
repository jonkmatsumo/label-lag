package main

import (
	"context"
	"database/sql"
	"flag"
	"log/slog"
	"os"

	"github.com/jonkmatsumo/label-lag/go/analytics/internal/config"
	_ "github.com/lib/pq"
)

func main() {
	reset := flag.Bool("reset", false, "Reset aggregates before backfilling")
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

	if *reset {
		slog.Info("resetting aggregates")
		_, err = db.ExecContext(ctx, "TRUNCATE TABLE aggregates_daily, aggregates_hourly")
		if err != nil {
			slog.Error("failed to truncate aggregates", "error", err)
			os.Exit(1)
		}
	}

	slog.Info("starting backfill")

	backfillSQLDaily := `
		INSERT INTO aggregates_daily (tenant_id, date, total_decisions, total_alerts, sum_score, rules_fired_total)
		SELECT
			'',
			DATE(ts),
			COUNT(*),
			SUM(CASE WHEN decision IN ('REVIEW', 'REJECT') THEN 1 ELSE 0 END),
			SUM(final_score),
			SUM(COALESCE(jsonb_array_length(rule_impacts), 0))
		FROM inference_events
		GROUP BY 1, 2
		ON CONFLICT (tenant_id, date) DO UPDATE SET
			total_decisions = EXCLUDED.total_decisions,
			total_alerts = EXCLUDED.total_alerts,
			sum_score = EXCLUDED.sum_score,
			rules_fired_total = EXCLUDED.rules_fired_total
	`

	backfillSQLHourly := `
		INSERT INTO aggregates_hourly (tenant_id, hour, total_decisions, total_alerts, sum_score, rules_fired_total)
		SELECT
			'',
			date_trunc('hour', ts),
			COUNT(*),
			SUM(CASE WHEN decision IN ('REVIEW', 'REJECT') THEN 1 ELSE 0 END),
			SUM(final_score),
			SUM(COALESCE(jsonb_array_length(rule_impacts), 0))
		FROM inference_events
		GROUP BY 1, 2
		ON CONFLICT (tenant_id, hour) DO UPDATE SET
			total_decisions = EXCLUDED.total_decisions,
			total_alerts = EXCLUDED.total_alerts,
			sum_score = EXCLUDED.sum_score,
			rules_fired_total = EXCLUDED.rules_fired_total
	`

	_, err = db.ExecContext(ctx, backfillSQLDaily)
	if err != nil {
		slog.Error("failed to backfill daily aggregates", "error", err)
		os.Exit(1)
	}
	slog.Info("daily aggregates backfilled")

	_, err = db.ExecContext(ctx, backfillSQLHourly)
	if err != nil {
		slog.Error("failed to backfill hourly aggregates", "error", err)
		os.Exit(1)
	}
	slog.Info("hourly aggregates backfilled")

	slog.Info("backfill complete")
}
