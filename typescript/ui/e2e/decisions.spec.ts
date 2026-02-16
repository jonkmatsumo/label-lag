import { test, expect } from '@playwright/test';

test.describe('Decisions Page', () => {
    test.beforeEach(async ({ page }) => {
        // Mock the decisions list API
        await page.route('**/bff/v1/decisions*', async (route) => {
            await route.fulfill({
                json: {
                    decisions: [
                        {
                            decision_id: 'dec-123',
                            user_id: 'user-1',
                            decision: 'APPROVE',
                            score: 0.95,
                            timestamp: new Date().toISOString(),
                            model_version: 'v1.0.0',
                        },
                        {
                            decision_id: 'dec-456',
                            user_id: 'user-2',
                            decision: 'DECLINE',
                            score: 0.15,
                            timestamp: new Date().toISOString(),
                            model_version: 'v1.0.0',
                        },
                    ],
                    pagination: {
                        total: 2,
                        next_cursor: null,
                    },
                },
            });
        });

        await page.goto('/decisions');
    });

    test('displays decisions list', async ({ page }) => {
        await expect(page.getByRole('heading', { name: 'Decisions', level: 1 })).toBeVisible();
        const table = page.locator('table');
        await expect(table).toBeVisible();
        await expect(table.getByText('user-1')).toBeVisible();
        await expect(table.getByText('APPROVE')).toBeVisible();
    });

    test('filters by decision type', async ({ page }) => {
        // Verify filter input exists
        const select = page.getByRole('combobox').filter({ hasText: 'All Decisions' });
        await expect(select).toBeVisible();

        // We mock the filtered request to verify it's triggered
        let filteredRequestMade = false;
        await page.route('**/bff/v1/decisions?*decision=DECLINE*', async (route) => {
            filteredRequestMade = true;
            await route.fulfill({
                json: {
                    decisions: [],
                    pagination: { total: 0 },
                },
            });
        });

        // Interact with filter (using class selector as fallback if role is ambiguous)
        await page.locator('select.form-select').selectOption('DECLINE');

        // Wait for the request to be captured
        await expect.poll(() => filteredRequestMade).toBeTruthy();
    });
});
