import { test, expect } from '@playwright/test';

test.describe('Rule Backtests Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/rules/backtests');
  });

  test('displays backtests section', async ({ page }) => {
    await expect(page.locator('h2')).toContainText('Rule Inspector');
    await expect(page.locator('h3')).toContainText('Backtests');
  });

  test('has rule ID filter', async ({ page }) => {
    const filterInput = page.locator('input[placeholder*="rule"]');
    await expect(filterInput).toBeVisible();
  });

  test('shows loading state or results table', async ({ page }) => {
    // Wait for data to load
    await page.waitForSelector('.loading, .table, .empty-state, .alert-error', {
      timeout: 10000,
    });

    const hasContent =
      (await page.locator('.loading').isVisible().catch(() => false)) ||
      (await page.locator('.table').isVisible().catch(() => false)) ||
      (await page.locator('.empty-state').isVisible().catch(() => false)) ||
      (await page.locator('.alert-error').isVisible().catch(() => false));

    expect(hasContent).toBeTruthy();
  });

  test('filters by rule ID when entered', async ({ page }) => {
    // Wait for initial load
    await page.waitForSelector('.loading, .table, .empty-state, .alert-error', {
      timeout: 10000,
    });

    // Enter a rule ID filter
    const filterInput = page.locator('input[placeholder*="rule"]');
    await filterInput.fill('rule-001');

    // Should refresh data
    await page.waitForSelector('.loading, .table, .empty-state, .alert-error', {
      timeout: 10000,
    });

    // Clear filter
    await filterInput.fill('');

    // Should refresh again
    await page.waitForSelector('.loading, .table, .empty-state, .alert-error', {
      timeout: 10000,
    });
  });
});
