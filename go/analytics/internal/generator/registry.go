package generator

import (
	"fmt"
	"os"

	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"gopkg.in/yaml.v3"
)

type GeneratorConfig struct {
	Generators []FeatureConfig `yaml:"generators"`
}

type FeatureConfig struct {
	Name    string                 `yaml:"name"`
	Enabled bool                   `yaml:"enabled"`
	Type    string                 `yaml:"type"` // "numeric" or "categorical"
	Params  map[string]interface{} `yaml:"params"`
}

type NumericGeneratorFunc func(rng *RNG, r *pb.GeneratedRecord, params map[string]interface{}) (float64, error)
type CategoricalGeneratorFunc func(rng *RNG, r *pb.GeneratedRecord, params map[string]interface{}) (string, error)

type GeneratorRegistry struct {
	numericGenerators     map[string]NumericGeneratorFunc
	categoricalGenerators map[string]CategoricalGeneratorFunc
	config                *GeneratorConfig
}

func NewGeneratorRegistry() *GeneratorRegistry {
	reg := &GeneratorRegistry{
		numericGenerators:     make(map[string]NumericGeneratorFunc),
		categoricalGenerators: make(map[string]CategoricalGeneratorFunc),
	}
	reg.registerDefaults()
	return reg
}

func (reg *GeneratorRegistry) registerDefaults() {
	reg.numericGenerators["experimental_velocity_7d"] = func(rng *RNG, r *pb.GeneratedRecord, params map[string]interface{}) (float64, error) {
		multiplier := 1.0
		if m, ok := params["multiplier"].(float64); ok {
			multiplier = m
		} else if m, ok := params["multiplier"].(int); ok {
			multiplier = float64(m)
		}
		return float64(r.BankConnectionsCount_7D) * multiplier, nil
	}

	reg.numericGenerators["experimental_balance_delta_abs"] = func(rng *RNG, r *pb.GeneratedRecord, params map[string]interface{}) (float64, error) {
		min := 0.0
		max := 100.0
		if val, ok := params["min"].(float64); ok {
			min = val
		} else if val, ok := params["min"].(int); ok {
			min = float64(val)
		}
		if val, ok := params["max"].(float64); ok {
			max = val
		} else if val, ok := params["max"].(int); ok {
			max = float64(val)
		}
		return rng.Float64Range(min, max), nil
	}

	reg.numericGenerators["amount_dyn"] = func(rng *RNG, r *pb.GeneratedRecord, params map[string]interface{}) (float64, error) {
		return r.Amount, nil
	}

	reg.categoricalGenerators["device_type"] = func(rng *RNG, r *pb.GeneratedRecord, params map[string]interface{}) (string, error) {
		values, ok := params["values"].([]interface{})
		if !ok || len(values) == 0 {
			return "unknown", nil
		}
		idx := rng.IntN(len(values))
		if s, ok := values[idx].(string); ok {
			return s, nil
		}
		return fmt.Sprintf("%v", values[idx]), nil
	}
}

func (reg *GeneratorRegistry) LoadConfig(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read generator config: %v", err)
	}

	var config GeneratorConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return fmt.Errorf("failed to unmarshal generator config: %v", err)
	}

	// Validate config
	for _, fc := range config.Generators {
		if fc.Type == "numeric" {
			if _, ok := reg.numericGenerators[fc.Name]; !ok {
				return fmt.Errorf("unknown numeric generator: %s", fc.Name)
			}
		} else if fc.Type == "categorical" {
			if _, ok := reg.categoricalGenerators[fc.Name]; !ok {
				return fmt.Errorf("unknown categorical generator: %s", fc.Name)
			}
		} else {
			return fmt.Errorf("unknown generator type: %s for %s", fc.Type, fc.Name)
		}
	}

	reg.config = &config
	return nil
}

func (reg *GeneratorRegistry) Populate(rng *RNG, r *pb.GeneratedRecord) {
	if r.NumericalFeatures == nil {
		r.NumericalFeatures = make(map[string]float64)
	}
	if r.CategoricalFeatures == nil {
		r.CategoricalFeatures = make(map[string]string)
	}

	if reg.config == nil {
		return
	}

	for _, fc := range reg.config.Generators {
		if !fc.Enabled {
			continue
		}

		if fc.Type == "numeric" {
			gen := reg.numericGenerators[fc.Name]
			val, err := gen(rng, r, fc.Params)
			if err == nil {
				r.NumericalFeatures[fc.Name] = val
			}
		} else if fc.Type == "categorical" {
			gen := reg.categoricalGenerators[fc.Name]
			val, err := gen(rng, r, fc.Params)
			if err == nil {
				r.CategoricalFeatures[fc.Name] = val
			}
		}
	}
}
