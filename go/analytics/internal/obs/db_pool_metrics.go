package obs

import (
	"database/sql"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// DBPoolStatsCollector exports sql.DB pool stats as Prometheus metrics.
type DBPoolStatsCollector struct {
	db *sql.DB
	mu sync.Mutex

	openConnectionsDesc    *prometheus.Desc
	inUseConnectionsDesc   *prometheus.Desc
	idleConnectionsDesc    *prometheus.Desc
	maxOpenConnectionsDesc *prometheus.Desc
	waitCountDesc          *prometheus.Desc
	waitDurationDesc       *prometheus.Desc
	maxIdleClosedDesc      *prometheus.Desc
	maxLifetimeClosedDesc  *prometheus.Desc
	acquireDurationHist    prometheus.Histogram

	lastWaitCount    int64
	lastWaitDuration time.Duration
}

func NewDBPoolStatsCollector(db *sql.DB) *DBPoolStatsCollector {
	return &DBPoolStatsCollector{
		db: db,
		openConnectionsDesc: prometheus.NewDesc(
			"analytics_db_pool_open_connections",
			"Current number of established database connections.",
			nil,
			nil,
		),
		inUseConnectionsDesc: prometheus.NewDesc(
			"analytics_db_pool_in_use_connections",
			"Current number of database connections in use.",
			nil,
			nil,
		),
		idleConnectionsDesc: prometheus.NewDesc(
			"analytics_db_pool_idle_connections",
			"Current number of idle database connections.",
			nil,
			nil,
		),
		maxOpenConnectionsDesc: prometheus.NewDesc(
			"analytics_db_pool_max_open_connections",
			"Configured maximum number of open database connections.",
			nil,
			nil,
		),
		waitCountDesc: prometheus.NewDesc(
			"analytics_db_pool_wait_count_total",
			"Total number of waits for a database connection.",
			nil,
			nil,
		),
		waitDurationDesc: prometheus.NewDesc(
			"analytics_db_pool_wait_duration_seconds_total",
			"Total time spent waiting for a database connection.",
			nil,
			nil,
		),
		maxIdleClosedDesc: prometheus.NewDesc(
			"analytics_db_pool_max_idle_closed_total",
			"Total number of connections closed due to max idle constraints.",
			nil,
			nil,
		),
		maxLifetimeClosedDesc: prometheus.NewDesc(
			"analytics_db_pool_max_lifetime_closed_total",
			"Total number of connections closed due to max lifetime constraints.",
			nil,
			nil,
		),
		acquireDurationHist: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "analytics_db_pool_acquire_duration_seconds",
			Help:    "Proxy histogram for DB connection acquisition wait duration based on incremental pool wait stats.",
			Buckets: []float64{.0005, .001, .0025, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5},
		}),
	}
}

func (c *DBPoolStatsCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.openConnectionsDesc
	ch <- c.inUseConnectionsDesc
	ch <- c.idleConnectionsDesc
	ch <- c.maxOpenConnectionsDesc
	ch <- c.waitCountDesc
	ch <- c.waitDurationDesc
	ch <- c.maxIdleClosedDesc
	ch <- c.maxLifetimeClosedDesc
	c.acquireDurationHist.Describe(ch)
}

func (c *DBPoolStatsCollector) Collect(ch chan<- prometheus.Metric) {
	if c.db == nil {
		return
	}
	stats := c.db.Stats()

	ch <- prometheus.MustNewConstMetric(c.openConnectionsDesc, prometheus.GaugeValue, float64(stats.OpenConnections))
	ch <- prometheus.MustNewConstMetric(c.inUseConnectionsDesc, prometheus.GaugeValue, float64(stats.InUse))
	ch <- prometheus.MustNewConstMetric(c.idleConnectionsDesc, prometheus.GaugeValue, float64(stats.Idle))
	ch <- prometheus.MustNewConstMetric(c.maxOpenConnectionsDesc, prometheus.GaugeValue, float64(stats.MaxOpenConnections))
	ch <- prometheus.MustNewConstMetric(c.waitCountDesc, prometheus.CounterValue, float64(stats.WaitCount))
	ch <- prometheus.MustNewConstMetric(c.waitDurationDesc, prometheus.CounterValue, stats.WaitDuration.Seconds())
	ch <- prometheus.MustNewConstMetric(c.maxIdleClosedDesc, prometheus.CounterValue, float64(stats.MaxIdleClosed))
	ch <- prometheus.MustNewConstMetric(c.maxLifetimeClosedDesc, prometheus.CounterValue, float64(stats.MaxLifetimeClosed))

	c.mu.Lock()
	deltaWaitCount := stats.WaitCount - c.lastWaitCount
	deltaWaitDuration := stats.WaitDuration - c.lastWaitDuration
	c.lastWaitCount = stats.WaitCount
	c.lastWaitDuration = stats.WaitDuration
	c.mu.Unlock()

	if deltaWaitCount > 0 && deltaWaitDuration > 0 {
		c.acquireDurationHist.Observe(deltaWaitDuration.Seconds() / float64(deltaWaitCount))
	}
	c.acquireDurationHist.Collect(ch)
}

func RegisterDBPoolStatsCollector(reg prometheus.Registerer, db *sql.DB) error {
	if db == nil {
		return fmt.Errorf("db is required")
	}
	if reg == nil {
		reg = prometheus.DefaultRegisterer
	}
	collector := NewDBPoolStatsCollector(db)
	if err := reg.Register(collector); err != nil {
		var alreadyRegistered prometheus.AlreadyRegisteredError
		if errors.As(err, &alreadyRegistered) {
			return nil
		}
		return err
	}
	return nil
}
