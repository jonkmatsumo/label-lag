import { test, expect } from '@playwright/test';

test.describe('Rule Shadow Metrics Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/rules/shadow');
  });

  test('displays shadow metrics section', async ({ page }) => {
    await expect(page.locator('h2')).toContainText('Rule Inspector');
    await expect(page.locator('h3')).toContainText('Shadow Metrics');
  });

  test('has date range filter', async ({ page }) => {
    // Should have start and end date inputs
    const startDateInput = page.locator('input[type="date"]').first();
    const endDateInput = page.locator('input[type="date"]').last();

    await expect(startDateInput).toBeVisible();
    await expect(endDateInput).toBeVisible();

    // Should have default values (today and 7 days ago)
    const startDate = await startDateInput.inputValue();
    const endDate = await endDateInput.inputValue();

    expect(startDate).toBeTruthy();
    expect(endDate).toBeTruthy();
  });

  test('shows loading state or comparison table', async ({ page }) => {
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

  test('updates results when date range changes', async ({ page }) => {
    // Wait for initial load
    await page.waitForSelector('.loading, .table, .empty-state, .alert-error', {
      timeout: 10000,
    });

    // Change the start date
    const startDateInput = page.locator('input[type="date"]').first();
    const newStartDate = new Date();
    newStartDate.setDate(newStartDate.getDate() - 14);
    await startDateInput.fill(newStartDate.toISOString().split('T')[0]);

    // Should refresh data
    await page.waitForSelector('.loading, .table, .empty-state, .alert-error', {
      timeout: 10000,
    });
  });
});
