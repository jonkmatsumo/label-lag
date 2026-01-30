package db

import (
	"context"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/stretchr/testify/assert"
)

func TestValidateSchema(t *testing.T) {
	db, mock, err := sqlmock.New()
	assert.NoError(t, err)
	defer db.Close()

	ctx := context.Background()

	t.Run("valid schema", func(t *testing.T) {
		expected := []TableSchema{
			{Name: "table1", Columns: []string{"col1", "col2"}},
		}

		mock.ExpectQuery("SELECT EXISTS.*information_schema.tables").
			WithArgs("table1").
			WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(true))

		mock.ExpectQuery("SELECT EXISTS.*information_schema.columns").
			WithArgs("table1", "col1").
			WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(true))

		mock.ExpectQuery("SELECT EXISTS.*information_schema.columns").
			WithArgs("table1", "col2").
			WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(true))

		err := ValidateSchema(ctx, db, expected)
		assert.NoError(t, err)
	})

	t.Run("missing table", func(t *testing.T) {
		expected := []TableSchema{
			{Name: "table1", Columns: []string{"col1"}},
		}

		mock.ExpectQuery("SELECT EXISTS.*information_schema.tables").
			WithArgs("table1").
			WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(false))

		err := ValidateSchema(ctx, db, expected)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "required table table1 is missing")
	})

	t.Run("missing column", func(t *testing.T) {
		expected := []TableSchema{
			{Name: "table1", Columns: []string{"col1"}},
		}

		mock.ExpectQuery("SELECT EXISTS.*information_schema.tables").
			WithArgs("table1").
			WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(true))

		mock.ExpectQuery("SELECT EXISTS.*information_schema.columns").
			WithArgs("table1", "col1").
			WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(false))

		err := ValidateSchema(ctx, db, expected)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "required column table1.col1 is missing")
	})
}
