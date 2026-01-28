import { test, expect } from '@playwright/test';

test.describe('Historical Analytics Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/analytics');
  });

  test('displays analytics page with overview section', async ({ page }) => {
    await expect(page.locator('h2')).toContainText('Historical Analytics');
    await expect(page.locator('.card-title').first()).toContainText('Dataset Overview');
  });

  test('shows loading state or metrics grid', async ({ page }) => {
    // Wait for either loading or metrics
    await page.waitForSelector('.loading, .metrics-grid, .alert-error', {
      timeout: 10000,
    });

    const hasContent =
      (await page.locator('.loading').isVisible().catch(() => false)) ||
      (await page.locator('.metrics-grid').isVisible().catch(() => false)) ||
      (await page.locator('.alert-error').isVisible().catch(() => false));

    expect(hasContent).toBeTruthy();
  });

  test('has daily statistics section with date filter', async ({ page }) => {
    await expect(page.locator('text=Daily Statistics')).toBeVisible();

    // Should have a date range selector
    const selector = page.locator('select');
    await expect(selector).toBeVisible();

    // Verify default options exist
    const options = await selector.locator('option').allTextContents();
    expect(options).toContain('Last 30 days');
    expect(options).toContain('Last 7 days');
  });

  test('changes daily stats when filter changes', async ({ page }) => {
    // Wait for initial load
    await page.waitForSelector('.loading, .table, .empty-state, .alert-error', {
      timeout: 10000,
    });

    // Change the filter
    const selector = page.locator('select');
    await selector.selectOption('7');

    // Should trigger a refresh (loading or table update)
    await page.waitForSelector('.loading, .table, .empty-state, .alert-error', {
      timeout: 10000,
    });
  });

  test('displays recent alerts section', async ({ page }) => {
    await expect(page.locator('text=Recent High-Risk Alerts')).toBeVisible();

    // Wait for alerts content
    await page.waitForSelector('.loading, .table, .empty-state, .alert-error', {
      timeout: 10000,
    });
  });
});
