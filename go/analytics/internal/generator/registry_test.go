package generator

import (
	"os"
	"testing"

	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
)

func TestGeneratorRegistry_Populate(t *testing.T) {
	reg := NewGeneratorRegistry()

	// Create a temp config file
	configContent := `
generators:
  - name: "experimental_velocity_7d"
    enabled: true
    type: "numeric"
    params:
      multiplier: 2.0
  - name: "device_type"
    enabled: true
    type: "categorical"
    params:
      values: ["test_device"]
`
	tmpfile, err := os.CreateTemp("", "generator_config_*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpfile.Name())

	if _, err := tmpfile.Write([]byte(configContent)); err != nil {
		t.Fatal(err)
	}
	if err := tmpfile.Close(); err != nil {
		t.Fatal(err)
	}

	if err := reg.LoadConfig(tmpfile.Name()); err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	seed := int64(42)
	rng := NewRNG(&seed)
	r := &pb.GeneratedRecord{
		BankConnectionsCount_7D: 10,
	}

	reg.Populate(rng, r)

	if val, ok := r.NumericalFeatures["experimental_velocity_7d"]; !ok || val != 20.0 {
		t.Errorf("Expected experimental_velocity_7d to be 20.0, got %f", val)
	}

	if val, ok := r.CategoricalFeatures["device_type"]; !ok || val != "test_device" {
		t.Errorf("Expected device_type to be 'test_device', got %s", val)
	}
}

func TestGeneratorRegistry_Validation(t *testing.T) {
	reg := NewGeneratorRegistry()

	configContent := `
generators:
  - name: "unknown_generator"
    enabled: true
    type: "numeric"
    params: {}
`
	tmpfile, err := os.CreateTemp("", "generator_config_*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpfile.Name())

	if _, err := tmpfile.Write([]byte(configContent)); err != nil {
		t.Fatal(err)
	}
	tmpfile.Close()

	if err := reg.LoadConfig(tmpfile.Name()); err == nil {
		t.Error("Expected error for unknown generator, got nil")
	}
}
