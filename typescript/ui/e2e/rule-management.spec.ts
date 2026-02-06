import { test, expect } from '@playwright/test';

test.describe('Rule Management Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/rules');
  });

  test('displays rule management section', async ({ page }) => {
    await expect(page.locator('h2')).toContainText('Rule Inspector');
    await expect(page.locator('h3')).toContainText('Draft Rules');
  });

  test('shows tabs for different rule sections', async ({ page }) => {
    await expect(page.locator('.tab:has-text("Management")')).toBeVisible();
    await expect(page.locator('.tab:has-text("Sandbox")')).toBeVisible();
    await expect(page.locator('.tab:has-text("Shadow Metrics")')).toBeVisible();
    await expect(page.locator('.tab:has-text("Backtests")')).toBeVisible();
    await expect(page.locator('.tab:has-text("Suggestions")')).toBeVisible();
  });

  test('navigates between tabs', async ({ page }) => {
    // Click Sandbox tab
    await page.click('.tab:has-text("Sandbox")');
    await expect(page).toHaveURL(/\/rules\/sandbox/);
    await expect(page.locator('h3')).toContainText('Sandbox Evaluation');

    // Click Shadow tab
    await page.click('.tab:has-text("Shadow Metrics")');
    await expect(page).toHaveURL(/\/rules\/shadow/);
    await expect(page.locator('h3')).toContainText('Shadow Metrics');

    // Click back to Management
    await page.click('.tab:has-text("Management")');
    await expect(page).toHaveURL(/\/rules$/);
    await expect(page.locator('h3')).toContainText('Draft Rules');
  });

  test('shows loading state or rules table', async ({ page }) => {
    // Wait for either loading, table, empty state, or error
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
});
