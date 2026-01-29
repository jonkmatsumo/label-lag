import { test, expect } from '@playwright/test';

// This test suite expects BFF to be running with BFF_INFERENCE_MODE=gateway
// It validates that the live scoring flow works under this configuration.

test.describe('Go Inference Gateway Integration', () => {
  test('should successfully score a transaction via Gateway', async ({ page }) => {
    await page.goto('/');
    
    // Fill transaction form
    await page.fill('input[value="user_001"]', 'go_user_test');
    await page.fill('input[value="100.00"]', '150.00');
    
    // Click Analyze
    await page.click('button:has-text("Analyze Risk")');
    
    // Expect score result
    await expect(page.locator('text=Score:')).toBeVisible({ timeout: 10000 });
    
    // Verify some risk factors appear (or empty list if safe)
    // We just check that the UI transitioned to results state
    await expect(page.locator('.score-gauge')).toBeVisible();
  });
});
