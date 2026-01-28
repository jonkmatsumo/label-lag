import { test, expect } from '@playwright/test';

test.describe('Model Registry', () => {
  test('should list model versions', async ({ page }) => {
    await page.route('**/bff/v1/mlflow/model-versions/search*', async route => {
      await route.fulfill({ json: { model_versions: [
        { version: "2", current_stage: "Production", run_id: "run1", creation_timestamp: Date.now(), status: "READY" },
        { version: "1", current_stage: "None", run_id: "run2", creation_timestamp: Date.now(), status: "READY" }
      ]}});
    });

    await page.goto('/model-lab');
    await page.getByRole('button', { name: 'Model Registry' }).click();
    
    await expect(page.getByText('v2')).toBeVisible();
    await expect(page.getByText('PRODUCTION')).toBeVisible();
    await expect(page.getByText('v1')).toBeVisible();
  });

  test('should expand version details', async ({ page }) => {
    await page.route('**/bff/v1/mlflow/model-versions/search*', async route => {
      await route.fulfill({ json: { model_versions: [
        { version: "1", current_stage: "Production", run_id: "abc", creation_timestamp: Date.now(), status: "READY" }
      ]}});
    });

    await page.route('**/bff/v1/mlflow/runs/abc/artifacts?path=cv_fold_metrics.json', async route => {
      await route.fulfill({ json: { precision: [0.9, 0.92] }});
    });
    
    await page.route('**/bff/v1/mlflow/runs/abc/artifacts?path=tuning_trials.json', async route => {
      await route.fulfill({ json: [{ trial: 1, value: 0.85, state: "COMPLETE" }]});
    });

    await page.route('**/bff/v1/mlflow/runs/abc/artifacts?path=split_manifest.json', async route => {
      await route.fulfill({ json: { train_size: 1000 }});
    });

    await page.goto('/model-lab');
    await page.getByRole('button', { name: 'Model Registry' }).click();
    
    // Click row to expand
    await page.getByText('v1').click();
    
    await expect(page.getByText('Cross-Validation Metrics')).toBeVisible();
    await expect(page.getByText('Tuning Trials')).toBeVisible();
    await expect(page.getByText('#1')).toBeVisible();
  });
});
