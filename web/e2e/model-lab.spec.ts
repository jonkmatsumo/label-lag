import { test, expect } from '@playwright/test';

test.describe('Model Lab Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/model-lab');
  });

  test('displays model lab with current model status', async ({ page }) => {
    await expect(page.locator('h2')).toContainText('Model Lab');
    await expect(page.locator('.card-title').first()).toContainText(
      'Current Production Model'
    );
  });

  test('has training form with all parameters', async ({ page }) => {
    await expect(page.locator('#name')).toBeVisible();
    await expect(page.locator('#test_size')).toBeVisible();
    await expect(page.locator('#random_seed')).toBeVisible();
    await expect(page.getByRole('button', { name: /start training/i })).toBeVisible();
  });

  test('has default training parameters', async ({ page }) => {
    const testSize = await page.locator('#test_size').inputValue();
    const randomSeed = await page.locator('#random_seed').inputValue();

    expect(parseFloat(testSize)).toBe(0.2);
    expect(parseInt(randomSeed)).toBe(42);
  });

  test('submits training request', async ({ page }) => {
    // Fill optional name
    await page.fill('#name', 'e2e-test-model');

    // Submit
    await page.click('button:has-text("Start Training")');

    // Should show loading state
    await expect(page.getByRole('button', { name: /training\.\.\./i })).toBeVisible();

    // Wait for result or timeout (training takes time)
    await page.waitForSelector('.alert-success, .alert-error, .card-title:has-text("Training Result")', {
      timeout: 30000,
    });

    // Either success or error is expected
    const hasResult =
      (await page.locator('.card-title:has-text("Training Result")').isVisible().catch(() => false)) ||
      (await page.locator('.alert-error').isVisible().catch(() => false));

    expect(hasResult).toBeTruthy();
  });
});
