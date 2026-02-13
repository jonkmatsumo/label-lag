package main

import (
	"context"
	"database/sql"
	"flag"
	"log"
	"os"
	"time"

	"github.com/jonkmatsumo/label-lag/go/analytics/internal/config"
	"github.com/jonkmatsumo/label-lag/go/analytics/internal/store"
	_ "github.com/lib/pq"
)

func main() {
	tenantID := flag.String("tenant-id", defaultTenantID(), "Tenant ID to prune dataset profiles for")
	allTenants := flag.Bool("all-tenants", false, "Prune dataset profiles across all tenants")
	retentionDays := flag.Int("retention-days", 30, "Retention period in days for dataset profiles")
	flag.Parse()

	if *retentionDays <= 0 {
		log.Fatalf("retention-days must be > 0 (got %d)", *retentionDays)
	}
	if *allTenants && *tenantID != "" {
		log.Printf("ignoring --tenant-id=%q because --all-tenants was provided", *tenantID)
	}
	if !*allTenants && *tenantID == "" {
		log.Fatal("tenant-id is required unless --all-tenants is set")
	}

	dbURL, err := config.ResolveDatabaseURL(os.Getenv)
	if err != nil {
		log.Fatalf("failed to resolve database url: %v", err)
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

	retention := time.Duration(*retentionDays) * 24 * time.Hour
	olderThan := time.Now().Add(-retention)
	if *allTenants {
		log.Printf("WARNING: pruning dataset profiles for ALL tenants older than %d days (%s)", *retentionDays, olderThan.Format(time.RFC3339))
	} else {
		log.Printf("Pruning dataset profiles for tenant %q older than %d days (%s)", *tenantID, *retentionDays, olderThan.Format(time.RFC3339))
	}

	scopeTenantID := ""
	if !*allTenants {
		scopeTenantID = *tenantID
	}
	count, err := s.PruneDatasetProfilesByTenant(context.Background(), olderThan, scopeTenantID)
	if err != nil {
		log.Fatalf("pruning failed: %v", err)
	}

	if *allTenants {
		log.Printf("Successfully pruned %d dataset profiles across all tenants", count)
	} else {
		log.Printf("Successfully pruned %d dataset profiles for tenant %q", count, *tenantID)
	}
}

func defaultTenantID() string {
	if v := os.Getenv("TENANT_ID"); v != "" {
		return v
	}
	return "tenant-1"
}
