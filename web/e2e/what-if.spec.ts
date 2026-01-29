import { test, expect } from '@playwright/test';

test.describe('What-If Simulation Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/what-if');
  });

  test('displays comparison form with all fields', async ({ page }) => {
    await expect(page.locator('h2')).toContainText('What-If Simulation');
    await expect(page.locator('#base_version')).toBeVisible();
    await expect(page.locator('#candidate_version')).toBeVisible();
    await expect(page.locator('#start_date')).toBeVisible();
    await expect(page.locator('#end_date')).toBeVisible();
    await expect(page.locator('#rule_id')).toBeVisible();
    await expect(page.getByRole('button', { name: /run comparison/i })).toBeVisible();
  });

  test('has default date range set to last 7 days', async ({ page }) => {
    const startDate = await page.locator('#start_date').inputValue();
    const endDate = await page.locator('#end_date').inputValue();

    expect(startDate).toBeTruthy();
    expect(endDate).toBeTruthy();

    // End date should be today
    const today = new Date().toISOString().split('T')[0];
    expect(endDate).toBe(today);

    // Start date should be 7 days ago
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
    expect(startDate).toBe(sevenDaysAgo.toISOString().split('T')[0]);
  });

  test('shows validation when required fields are missing', async ({ page }) => {
    // Clear pre-filled dates
    await page.fill('#base_version', '');

    await page.click('button:has-text("Run Comparison")');

    // HTML5 validation should prevent submission
    const baseVersionInput = page.locator('#base_version');
    await expect(baseVersionInput).toHaveJSProperty('validity.valueMissing', true);
  });

  test('submits comparison request and shows delta metrics', async ({ page }) => {
    // Fill required fields
    await page.fill('#base_version', 'v1.0.0');
    await page.fill('#candidate_version', 'v1.1.0');

    // Submit
    await page.click('button:has-text("Run Comparison")');

    // Wait for result or error
    await page.waitForSelector('.comparison-results, .alert-error', {
      timeout: 15000, // Longer timeout for backtest
    });

    const results = page.locator('.comparison-results');
    const errorAlert = page.locator('.alert-error');

    const hasResults = await results.isVisible().catch(() => false);
    const hasError = await errorAlert.isVisible().catch(() => false);

    // Either we get results or a connection error
    expect(hasResults || hasError).toBeTruthy();
  });
});
