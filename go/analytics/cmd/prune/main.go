package main

import (
	"context"
	"database/sql"
	"flag"
	"log"
	"os"
	"time"

	"github.com/jonkmatsumo/label-lag/go/analytics/internal/store"
	_ "github.com/lib/pq"
)

func main() {
	retention := flag.Duration("retention", 30*24*time.Hour, "Retention period for dataset profiles (e.g., 720h)")
	flag.Parse()

	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgres://synthetic:synthetic_dev_password@localhost:5432/synthetic_data?sslmode=disable"
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		log.Fatalf("failed to connect to db: %v", err)
	}
	defer db.Close()

	if err := db.Ping(); err != nil {
		log.Fatalf("failed to ping db: %v", err)
	}

	s := store.NewSQLStore(db)

	olderThan := time.Now().Add(-*retention)
	log.Printf("Pruning dataset profiles older than %v (%s)", *retention, olderThan.Format(time.RFC3339))

	count, err := s.PruneDatasetProfiles(context.Background(), olderThan)
	if err != nil {
		log.Fatalf("pruning failed: %v", err)
	}

	log.Printf("Successfully pruned %d dataset profiles", count)
}
