package db

import (
	"fmt"
	"strings"
)

// QueryBuilder logic to construct dynamic SQL
type QueryBuilder struct {
	baseQuery  string
	conditions []string
	args       []interface{}
	orderBy    string
	limit      int32
	offset     int32
}

func NewQueryBuilder(baseQuery string) *QueryBuilder {
	return &QueryBuilder{
		baseQuery:  baseQuery,
		conditions: []string{},
		args:       []interface{}{},
	}
}

func (qb *QueryBuilder) AddCondition(condition string, args ...interface{}) {
	qb.conditions = append(qb.conditions, condition)
	qb.args = append(qb.args, args...)
}

func (qb *QueryBuilder) AddOrderBy(orderBy string) {
	qb.orderBy = orderBy
}

func (qb *QueryBuilder) SetLimit(limit int32) {
	qb.limit = limit
}

func (qb *QueryBuilder) SetOffset(offset int32) {
	qb.offset = offset
}

func (qb *QueryBuilder) BuildCount() (string, []interface{}) {
	query := fmt.Sprintf("SELECT COUNT(*) FROM (%s", qb.baseQuery)
	if len(qb.conditions) > 0 {
		query += " WHERE " + strings.Join(qb.conditions, " AND ")
	}
	query += ") as count_query"
	return qb.replacePlaceholders(query), qb.args
}

func (qb *QueryBuilder) BuildSelect() (string, []interface{}) {
	query := qb.baseQuery
	if len(qb.conditions) > 0 {
		query += " WHERE " + strings.Join(qb.conditions, " AND ")
	}
	if qb.orderBy != "" {
		query += " ORDER BY " + qb.orderBy
	}
	if qb.limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", qb.limit)
	}
	if qb.offset > 0 {
		query += fmt.Sprintf(" OFFSET %d", qb.offset)
	}
	return qb.replacePlaceholders(query), qb.args
}

func (qb *QueryBuilder) replacePlaceholders(query string) string {
	// Simple placeholder replacement for $1, $2, etc.
	// This assumes that the base query doesn't have ? placeholders or that we want to replace all ?
	// with $n.
	// Note: stored procedures/complex queries might need more robust handling.
	n := 1
	for strings.Contains(query, "?") {
		query = strings.Replace(query, "?", fmt.Sprintf("$%d", n), 1)
		n++
	}
	return query
}
