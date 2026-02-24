package store

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"
)

type decisionCursor struct {
	CreatedAt time.Time `json:"created_at"`
	RequestId string    `json:"request_id"`
}

func encodeDecisionCursor(createdAt time.Time, requestID string) string {
	c := decisionCursor{
		CreatedAt: createdAt,
		RequestId: requestID,
	}
	b, _ := json.Marshal(c)
	return base64.StdEncoding.EncodeToString(b)
}

func decodeDecisionCursor(cursorStr string) (*decisionCursor, error) {
	if cursorStr == "" {
		return nil, nil
	}
	b, err := base64.StdEncoding.DecodeString(cursorStr)
	if err != nil {
		return nil, fmt.Errorf("invalid cursor encoding: %w", err)
	}
	var c decisionCursor
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("invalid cursor payload: %w", err)
	}
	return &c, nil
}

type jobCursor struct {
	CreatedAt time.Time `json:"created_at"`
	JobId     string    `json:"job_id"`
}

func encodeJobCursor(createdAt time.Time, jobID string) string {
	c := jobCursor{
		CreatedAt: createdAt,
		JobId:     jobID,
	}
	b, _ := json.Marshal(c)
	return base64.StdEncoding.EncodeToString(b)
}

func decodeJobCursor(cursorStr string) (*jobCursor, error) {
	if cursorStr == "" {
		return nil, nil
	}
	b, err := base64.StdEncoding.DecodeString(cursorStr)
	if err != nil {
		return nil, fmt.Errorf("invalid cursor encoding: %w", err)
	}
	var c jobCursor
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("invalid cursor payload: %w", err)
	}
	return &c, nil
}

type profileCursor struct {
	ComputedAt time.Time `json:"computed_at"`
	ProfileId  string    `json:"profile_id"`
}

func encodeProfileCursor(computedAt time.Time, profileID string) string {
	c := profileCursor{
		ComputedAt: computedAt,
		ProfileId:  profileID,
	}
	b, _ := json.Marshal(c)
	return base64.StdEncoding.EncodeToString(b)
}

func decodeProfileCursor(cursorStr string) (*profileCursor, error) {
	if cursorStr == "" {
		return nil, nil
	}
	b, err := base64.StdEncoding.DecodeString(cursorStr)
	if err != nil {
		return nil, fmt.Errorf("invalid cursor encoding: %w", err)
	}
	var c profileCursor
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("invalid cursor payload: %w", err)
	}
	return &c, nil
}

type trainingRunCursor struct {
	StartedAt time.Time `json:"started_at"`
	RunId     string    `json:"run_id"`
}

func encodeTrainingRunCursor(startedAt time.Time, runID string) string {
	c := trainingRunCursor{
		StartedAt: startedAt,
		RunId:     runID,
	}
	b, _ := json.Marshal(c)
	return base64.StdEncoding.EncodeToString(b)
}

func decodeTrainingRunCursor(cursorStr string) (*trainingRunCursor, error) {
	if cursorStr == "" {
		return nil, nil
	}
	b, err := base64.StdEncoding.DecodeString(cursorStr)
	if err != nil {
		return nil, fmt.Errorf("invalid cursor encoding: %w", err)
	}
	var c trainingRunCursor
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("invalid cursor payload: %w", err)
	}
	return &c, nil
}

const (
	DefaultLimit = 100
	MaxLimit     = 500
)

type transactionCursor struct {
	CreatedAt time.Time `json:"created_at"`
	RecordId  string    `json:"record_id"`
}

func encodeTransactionCursor(createdAt time.Time, recordID string) string {
	c := transactionCursor{
		CreatedAt: createdAt,
		RecordId:  recordID,
	}
	b, _ := json.Marshal(c)
	return base64.StdEncoding.EncodeToString(b)
}

func decodeTransactionCursor(cursorStr string) (*transactionCursor, error) {
	if cursorStr == "" {
		return nil, nil
	}
	b, err := base64.StdEncoding.DecodeString(cursorStr)
	if err != nil {
		return nil, fmt.Errorf("invalid cursor encoding: %w", err)
	}
	var c transactionCursor
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("invalid cursor payload: %w", err)
	}
	if c.RecordId == "" {
		return nil, fmt.Errorf("invalid cursor payload: missing record_id")
	}
	if c.CreatedAt.IsZero() {
		return nil, fmt.Errorf("invalid cursor payload: missing created_at")
	}
	return &c, nil
}
