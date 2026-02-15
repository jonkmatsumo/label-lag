package store

import (
	"encoding/base64"
	"testing"
	"time"
)

func TestCursorHardening(t *testing.T) {
	t.Run("DecisionCursor", func(t *testing.T) {
		// Valid round-trip
		now := time.Now().Truncate(time.Microsecond).UTC()
		enc := encodeDecisionCursor(now, "req-123")
		dec, err := decodeDecisionCursor(enc)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !dec.CreatedAt.Equal(now) || dec.RequestId != "req-123" {
			t.Errorf("mismatch: %+v", dec)
		}

		// Invalid base64
		_, err = decodeDecisionCursor("not-base64-!!!")
		if err == nil {
			t.Error("expected error for invalid base64")
		}

		// Invalid JSON
		badJSON := base64.StdEncoding.EncodeToString([]byte(`{"created_at": "invalid-date"}`))
		_, err = decodeDecisionCursor(badJSON)
		if err == nil {
			t.Error("expected error for invalid JSON date")
		}

		// Random binary data
		randData := base64.StdEncoding.EncodeToString([]byte{0x00, 0x01, 0x02, 0x03})
		_, err = decodeDecisionCursor(randData)
		if err == nil {
			t.Error("expected error for random binary data")
		}
	})

	t.Run("JobCursor", func(t *testing.T) {
		now := time.Now().Truncate(time.Microsecond).UTC()
		enc := encodeJobCursor(now, "job-123")
		dec, err := decodeJobCursor(enc)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !dec.CreatedAt.Equal(now) || dec.JobId != "job-123" {
			t.Errorf("mismatch: %+v", dec)
		}
	})

	t.Run("ProfileCursor", func(t *testing.T) {
		now := time.Now().Truncate(time.Microsecond).UTC()
		enc := encodeProfileCursor(now, "prof-123")
		dec, err := decodeProfileCursor(enc)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !dec.ComputedAt.Equal(now) || dec.ProfileId != "prof-123" {
			t.Errorf("mismatch: %+v", dec)
		}
	})

	t.Run("TrainingRunCursor", func(t *testing.T) {
		now := time.Now().Truncate(time.Microsecond).UTC()
		enc := encodeTrainingRunCursor(now, "run-123")
		dec, err := decodeTrainingRunCursor(enc)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !dec.StartedAt.Equal(now) || dec.RunId != "run-123" {
			t.Errorf("mismatch: %+v", dec)
		}
	})
}
