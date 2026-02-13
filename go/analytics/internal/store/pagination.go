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
