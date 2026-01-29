import { test, expect } from '@playwright/test';

test.describe('Dataset Management', () => {
  test('should display overview metrics', async ({ page }) => {
    await page.route('**/bff/v1/dataset/overview', async route => {
      await route.fulfill({ json: { 
        total_records: 5000, 
        fraud_rate: 0.05,
        unique_users: 100,
        fraud_records: 250
      }});
    });
    
    await page.route('**/bff/v1/dataset/schema', async route => {
      await route.fulfill({ json: { columns: [] }});
    });

    await page.goto('/dataset');
    await expect(page.getByText('5,000')).toBeVisible();
    await expect(page.getByText('5.00%')).toBeVisible();
  });

  test('should allow generating data', async ({ page }) => {
    await page.route('**/bff/v1/dataset/overview', async route => {
      await route.fulfill({ json: { total_records: 0 }});
    });
    
    await page.route('**/bff/v1/dataset/generate', async route => {
      await route.fulfill({ json: { success: true, total_records: 100, fraud_records: 5 }});
    });

    await page.goto('/dataset');
    await page.getByRole('button', { name: 'Generate' }).click();
    
    // Fill form
    await page.getByRole('button', { name: 'Generate Data' }).click();
    
    await expect(page.getByText('Generation Complete!')).toBeVisible();
    await expect(page.getByText('Created 100 records')).toBeVisible();
  });

  test('should show diagnostics charts', async ({ page }) => {
    await page.route('**/bff/v1/dataset/sample*', async route => {
      await route.fulfill({ json: { samples: [
        { velocity_24h: 10, is_fraudulent: false },
        { velocity_24h: 50, is_fraudulent: true }
      ]}});
    });

    await page.goto('/dataset');
    await page.getByRole('button', { name: 'Diagnostics' }).click();
    
    await expect(page.getByText('Distribution: velocity_24h')).toBeVisible();
    await expect(page.locator('.recharts-responsive-container')).toBeVisible();
  });
});
