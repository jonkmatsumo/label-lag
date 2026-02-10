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
	batchSize := flag.Int("batch-size", 5000, "Number of events to process per batch")
	sleepMs := flag.Int("sleep-ms", 0, "Sleep between batches")
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

	var minID, maxID int64
	err = db.QueryRowContext(ctx, "SELECT COALESCE(MIN(id), 0), COALESCE(MAX(id), 0) FROM inference_events").Scan(&minID, &maxID)
	if err != nil {
		slog.Error("failed to get event range", "error", err)
		os.Exit(1)
	}

	if maxID == 0 {
		slog.Info("no events to process")
		return
	}

	slog.Info("starting backfill", "min_id", minID, "max_id", maxID, "batch_size", *batchSize)

	for currentID := minID; currentID <= maxID; currentID += int64(*batchSize) {
		upperID := currentID + int64(*batchSize) - 1
		if upperID > maxID {
			upperID = maxID
		}

		slog.Info("processing batch", "from_id", currentID, "to_id", upperID)

		err = processBatch(ctx, db, currentID, upperID)
		if err != nil {
			slog.Error("failed to process batch", "from_id", currentID, "error", err)
			os.Exit(1)
		}

		if *sleepMs > 0 {
			time.Sleep(time.Duration(*sleepMs) * time.Millisecond)
		}
	}

	slog.Info("backfill complete")
}

func processBatch(ctx context.Context, db *sql.DB, fromID, toID int64) error {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

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
		WHERE id BETWEEN $1 AND $2
		GROUP BY 1, 2
		ON CONFLICT (tenant_id, date) DO UPDATE SET
			total_decisions = aggregates_daily.total_decisions + EXCLUDED.total_decisions,
			total_alerts = aggregates_daily.total_alerts + EXCLUDED.total_alerts,
			sum_score = aggregates_daily.sum_score + EXCLUDED.sum_score,
			rules_fired_total = aggregates_daily.rules_fired_total + EXCLUDED.rules_fired_total
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
		WHERE id BETWEEN $1 AND $2
		GROUP BY 1, 2
		ON CONFLICT (tenant_id, hour) DO UPDATE SET
			total_decisions = aggregates_hourly.total_decisions + EXCLUDED.total_decisions,
			total_alerts = aggregates_hourly.total_alerts + EXCLUDED.total_alerts,
			sum_score = aggregates_hourly.sum_score + EXCLUDED.sum_score,
			rules_fired_total = aggregates_hourly.rules_fired_total + EXCLUDED.rules_fired_total
	`

	_, err = tx.ExecContext(ctx, backfillSQLDaily, fromID, toID)
	if err != nil {
		return err
	}

	_, err = tx.ExecContext(ctx, backfillSQLHourly, fromID, toID)
	if err != nil {
		return err
	}

	return tx.Commit()
}
