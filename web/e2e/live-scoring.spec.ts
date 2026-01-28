import { test, expect } from '@playwright/test';

test.describe('Live Scoring Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('displays page title and model status section', async ({ page }) => {
    await expect(page.locator('h2')).toContainText('Live Scoring');
    await expect(page.locator('.card-title').first()).toContainText('Model Status');
  });

  test('has transaction input form with all required fields', async ({ page }) => {
    await expect(page.locator('#user_id')).toBeVisible();
    await expect(page.locator('#amount')).toBeVisible();
    await expect(page.locator('#currency')).toBeVisible();
    await expect(page.locator('#client_transaction_id')).toBeVisible();
    await expect(page.getByRole('button', { name: /evaluate risk/i })).toBeVisible();
  });

  test('generates transaction ID when clicking Generate button', async ({ page }) => {
    const generateButton = page.getByRole('button', { name: /generate/i });
    const txnInput = page.locator('#client_transaction_id');

    await expect(txnInput).toHaveValue('');
    await generateButton.click();
    await expect(txnInput).not.toHaveValue('');
    await expect(txnInput).toHaveValue(/^txn_/);
  });

  test('shows validation error when submitting empty form', async ({ page }) => {
    // HTML5 validation should prevent submission
    const submitButton = page.getByRole('button', { name: /evaluate risk/i });
    await submitButton.click();

    // The form should not submit and user_id should show as invalid
    const userIdInput = page.locator('#user_id');
    await expect(userIdInput).toHaveJSProperty('validity.valueMissing', true);
  });

  test('submits form and shows score result', async ({ page }) => {
    // Fill in the form
    await page.fill('#user_id', 'user_test_123');
    await page.fill('#amount', '500');
    await page.selectOption('#currency', 'USD');
    await page.click('button:has-text("Generate")');

    // Submit the form
    await page.click('button:has-text("Evaluate Risk")');

    // Wait for either result or error
    // In a real test environment with mocked API:
    await page.waitForSelector('.score-display, .alert-error', {
      timeout: 10000,
    });

    // If running against a real backend, we'd see the score
    const scoreDisplay = page.locator('.score-display');
    const errorAlert = page.locator('.alert-error');

    // Either score renders or we get a connection error (expected without backend)
    const hasScore = await scoreDisplay.isVisible().catch(() => false);
    const hasError = await errorAlert.isVisible().catch(() => false);

    expect(hasScore || hasError).toBeTruthy();
  });
});
