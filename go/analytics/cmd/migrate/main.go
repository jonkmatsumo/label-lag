package main

import (
	"database/sql"
	"log"
	"os"

	"github.com/jonkmatsumo/label-lag/go/analytics/internal/db"
	_ "github.com/lib/pq"
)

func main() {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgres://synthetic:synthetic_dev_password@localhost:5432/synthetic_data?sslmode=disable"
	}
	conn, err := sql.Open("postgres", dbURL)
	if err != nil {
		log.Fatalf("failed to connect to db: %v", err)
	}
	defer conn.Close()

	if err := db.InitDB(conn); err != nil {
		log.Fatalf("failed to initialize db: %v", err)
	}
	log.Println("Migrations applied successfully")
}
