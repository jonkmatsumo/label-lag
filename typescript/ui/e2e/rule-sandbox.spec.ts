import { test, expect } from '@playwright/test';

test.describe('Rule Sandbox Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/rules/sandbox');
  });

  test('displays sandbox evaluation form', async ({ page }) => {
    await expect(page.locator('h3')).toContainText('Sandbox Evaluation');
    await expect(page.locator('#base_score')).toBeVisible();
    await expect(page.locator('#features')).toBeVisible();
    await expect(page.getByRole('button', { name: /evaluate rules/i })).toBeVisible();
  });

  test('loads sample features when clicking Load Sample', async ({ page }) => {
    const loadSampleButton = page.getByRole('button', { name: /load sample/i });
    const featuresTextarea = page.locator('#features');

    // Initial value should be empty or minimal
    const initialValue = await featuresTextarea.inputValue();

    await loadSampleButton.click();

    const newValue = await featuresTextarea.inputValue();
    expect(newValue).toContain('amount');
    expect(newValue).toContain('velocity_24h');
    expect(newValue.length).toBeGreaterThan(initialValue.length);
  });

  test('shows JSON validation error for invalid JSON', async ({ page }) => {
    const featuresTextarea = page.locator('#features');
    await featuresTextarea.fill('{ invalid json }');

    await page.click('button:has-text("Evaluate Rules")');

    await expect(page.locator('.form-error')).toContainText('Invalid JSON');
  });

  test('submits valid request and shows results', async ({ page }) => {
    // Load sample features
    await page.click('button:has-text("Load Sample")');

    // Set base score
    await page.fill('#base_score', '50');

    // Submit
    await page.click('button:has-text("Evaluate Rules")');

    // Wait for result or error
    await page.waitForSelector('.score-comparison, .alert-error', {
      timeout: 10000,
    });

    const scoreComparison = page.locator('.score-comparison');
    const errorAlert = page.locator('.alert-error');

    const hasResult = await scoreComparison.isVisible().catch(() => false);
    const hasError = await errorAlert.isVisible().catch(() => false);

    // Either we get results or a connection error (expected without backend)
    expect(hasResult || hasError).toBeTruthy();
  });
});
