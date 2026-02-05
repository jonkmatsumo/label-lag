package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/rules"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	subcommand := os.Args[1]
	switch subcommand {
	case "validate":
		handleValidate(os.Args[2:])
	case "evaluate":
		handleEvaluate(os.Args[2:])
	case "diff":
		handleDiff(os.Args[2:])
	default:
		fmt.Printf("Unknown subcommand: %s\n", subcommand)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("Usage: rules-cli <subcommand> [flags]")
	fmt.Println("Subcommands:")
	fmt.Println("  validate --rules rules.json")
	fmt.Println("  evaluate --features features.json --base-score N --rules rules.json [--shadow] [--debug]")
	fmt.Println("  diff --features features.json --base-score N --a rules_a.json --b rules_b.json [--shadow] [--format json|table]")
}

func handleValidate(args []string) {
	fs := flag.NewFlagSet("validate", flag.ExitOnError)
	rulesPath := fs.String("rules", "", "Path to rules JSON file")
	_ = fs.Parse(args)

	if *rulesPath == "" {
		fmt.Println("Error: --rules is required")
		os.Exit(1)
	}

	rs, err := loadRuleSet(*rulesPath)
	if err != nil {
		fmt.Printf("Error loading rules: %v\n", err)
		os.Exit(1)
	}

	conflicts := rules.ValidateRuleset(rs)
	if len(conflicts) > 0 {
		fmt.Printf("Found %d conflicts:\n", len(conflicts))
		for _, c := range conflicts {
			fmt.Printf("[%s] %s: %s (Rules: %v)\n", c.Severity, c.Field, c.Details, c.RuleIDs)
		}
		os.Exit(1)
	}

	fmt.Println("Ruleset is valid and has no conflicts.")
}

func handleEvaluate(args []string) {
	fs := flag.NewFlagSet("evaluate", flag.ExitOnError)
	featuresPath := fs.String("features", "", "Path to features JSON file")
	baseScore := fs.Int("base-score", 50, "Baseline score")
	rulesPath := fs.String("rules", "", "Path to rules JSON file")
	shadow := fs.Bool("shadow", false, "Enable shadow mode (simulation)")
	debug := fs.Bool("debug", false, "Enable debug mode (per-rule timing)")
	_ = fs.Parse(args)

	if *featuresPath == "" || *rulesPath == "" {
		fmt.Println("Error: --features and --rules are required")
		os.Exit(1)
	}

	feats, err := loadFeatures(*featuresPath)
	if err != nil {
		fmt.Printf("Error loading features: %v\n", err)
		os.Exit(1)
	}

	rs, err := loadRuleSet(*rulesPath)
	if err != nil {
		fmt.Printf("Error loading rules: %v\n", err)
		os.Exit(1)
	}

	result, err := rules.EvaluateRules(feats, *baseScore, rs, rules.EvalOptions{Debug: *debug})
	if err != nil {
		fmt.Printf("Error evaluating rules: %v\n", err)
		os.Exit(1)
	}

	output := map[string]any{
		"final_score":          result.FinalScore,
		"rejected":             result.Rejected,
		"matched_rules":        result.MatchedRules,
		"shadow_matched_rules": result.ShadowMatchedRules,
		"evaluation_time_ms":   result.EvaluationTimeMS,
	}
	if *shadow {
		output["shadow_mode"] = true
	}
	if *debug {
		output["per_rule_timings_ms"] = result.PerRuleTimingsMS
	}

	data, _ := json.MarshalIndent(output, "", "  ")
	fmt.Println(string(data))
}

func handleDiff(args []string) {
	fs := flag.NewFlagSet("diff", flag.ExitOnError)
	featuresPath := fs.String("features", "", "Path to features JSON file")
	baseScore := fs.Int("base-score", 50, "Baseline score")
	pathA := fs.String("a", "", "Path to ruleset A")
	pathB := fs.String("b", "", "Path to ruleset B")
	format := fs.String("format", "json", "Output format (json|table)")
	_ = fs.Parse(args)

	if *featuresPath == "" || *pathA == "" || *pathB == "" {
		fmt.Println("Error: --features, --a, and --b are required")
		os.Exit(1)
	}

	feats, _ := loadFeatures(*featuresPath)
	rsA, _ := loadRuleSet(*pathA)
	rsB, _ := loadRuleSet(*pathB)

	resA, _ := rules.EvaluateRules(feats, *baseScore, rsA, rules.EvalOptions{})
	resB, _ := rules.EvaluateRules(feats, *baseScore, rsB, rules.EvalOptions{})

	// Simplified diff logic matching gateway
	diff := struct {
		Severity            string   `json:"severity"`
		ScoreDelta          int      `json:"score_delta"`
		MatchedRulesAdded   []string `json:"matched_rules_added"`
		MatchedRulesRemoved []string `json:"matched_rules_removed"`
	}{}

	diff.ScoreDelta = resB.FinalScore - resA.FinalScore
	
	mapA := make(map[string]bool)
	for _, r := range resA.MatchedRules {
		mapA[r] = true
	}
	mapB := make(map[string]bool)
	for _, r := range resB.MatchedRules {
		mapB[r] = true
	}
	for r := range mapB {
		if !mapA[r] {
			diff.MatchedRulesAdded = append(diff.MatchedRulesAdded, r)
		}
	}
	for r := range mapA {
		if !mapB[r] {
			diff.MatchedRulesRemoved = append(diff.MatchedRulesRemoved, r)
		}
	}

	// Severity
	diff.Severity = "cosmetic"
	if diff.ScoreDelta != 0 || len(diff.MatchedRulesAdded) > 0 || len(diff.MatchedRulesRemoved) > 0 {
		diff.Severity = "behavioral"
	}
	if resA.Rejected != resB.Rejected || absInt(diff.ScoreDelta) > 20 {
		diff.Severity = "breaking"
	}

	if *format == "table" {
		fmt.Printf("Severity:    %s\n", diff.Severity)
		fmt.Printf("Score Delta: %d\n", diff.ScoreDelta)
		fmt.Printf("Rules Added:   %v\n", diff.MatchedRulesAdded)
		fmt.Printf("Rules Removed: %v\n", diff.MatchedRulesRemoved)
	} else {
		data, _ := json.MarshalIndent(diff, "", "  ")
		fmt.Println(string(data))
	}
}

func loadRuleSet(path string) (rules.RuleSet, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return rules.RuleSet{}, err
	}
	var rs struct {
		Version string       `json:"version"`
		Rules   []rules.Rule `json:"rules"`
	}
	if err := json.Unmarshal(data, &rs); err != nil {
		return rules.RuleSet{}, err
	}
	return rules.RuleSet{Version: rs.Version, Rules: rs.Rules}, nil
}

func loadFeatures(path string) (map[string]any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var feats map[string]any
	if err := json.Unmarshal(data, &feats); err != nil {
		return nil, err
	}
	return feats, nil
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}